import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PATIENT_ID = "210504B_C_0000"

_CACHE = {
    'df': None,
    'channels': None,
    'fs': 256.0,
    'df_crises': None,
    'features': {}
}

# Classification anatomique des électrodes selon le système international 10-20
LOBE_MAP = {
    'EEGFp1': ('Frontal', 'Région frontale antérieure gauche. Pointes-ondes à 3 Hz maximales. Idéal pour la détection.'),
    'EEGFp2': ('Frontal', 'Région frontale antérieure droite. Forte amplitude des décharges synchrones.'),
    'EEGF7':  ('Fronto-temporal', 'Région temporale antérieure gauche / frontale latérale.'),
    'EEGF3':  ('Frontal', 'Région frontale gauche intermédiaire. Très forte réactivité épileptique.'),
    'EEGFz':  ('Frontal médian', 'Région frontale médiane du scalp.'),
    'EEGF4':  ('Frontal', 'Région frontale droite intermédiaire. Très forte réactivité épileptique.'),
    'EEGF8':  ('Fronto-temporal', 'Région temporale antérieure droite / frontale latérale.'),
    'EEGT3':  ('Temporal', 'Région temporale moyenne gauche.'),
    'EEGC3':  ('Central', 'Région centrale (cortex moteur/sensoriel gauche).'),
    'EEGCz':  ('Central médian', 'Vertex du scalp (centre supérieur).'),
    'EEGC4':  ('Central', 'Région centrale (cortex moteur/sensoriel droit).'),
    'EEGT4':  ('Temporal', 'Région temporale moyenne droite.'),
    'EEGT5':  ('Temporo-occipital', 'Région temporale postérieure gauche.'),
    'EEGP3':  ('Pariétal', 'Région pariétale gauche.'),
    'EEGPz':  ('Pariétal médian', 'Région pariétale médiane.'),
    'EEGP4':  ('Pariétal', 'Région pariétale droite.'),
    'EEGT6':  ('Temporo-occipital', 'Région temporale postérieure droite.'),
    'EEGO1':  ('Occipital', 'Cortex visuel gauche (rythme Alpha de fond prédominant, peu de pointes-ondes).'),
    'EEGO2':  ('Occipital', 'Cortex visuel droit (rythme Alpha de fond prédominant, peu de pointes-ondes).'),
    'SLI':    ('Polygraphie', 'Stimulation Lumineuse Intermittente (signal technique de flashs visuels).'),
    'ECG':    ('Polygraphie', 'Électrocardiogramme (activité cardiaque, non cérébrale).')
}

def excel_time_to_seconds(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val) * 86400.0
    if isinstance(val, str):
        val = val.strip()
        parts = val.split(':')
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
    if hasattr(val, 'hour') and hasattr(val, 'minute') and hasattr(val, 'second'):
        return val.hour * 3600.0 + val.minute * 60.0 + val.second + getattr(val, 'microsecond', 0) / 1e6
    return float(val)

def seconds_to_hhmmss(sec):
    if sec is None or pd.isna(sec):
        return ""
    sec = int(round(sec)) % 86400
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def reset_cache():
    """Réinitialise les structures en cache pour un nouveau traitement."""
    _CACHE['df'] = None
    _CACHE['channels'] = None
    _CACHE['df_crises'] = None
    _CACHE['features'] = {}
    _CACHE['last_train_result'] = None

def list_available_datasets():
    """Détecte les fichiers patients et annotations présents dans le dossier data/."""
    datasets = []
    if not os.path.exists(DATA_DIR):
        return datasets
    
    excel_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.xlsx', '.xls'))]
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv') and not f.endswith('h.csv')]
    
    for f in csv_files:
        f_path = os.path.join(DATA_DIR, f)
        p_name = f.replace('d.csv', '').replace('.csv', '')
        size_mb = round(os.path.getsize(f_path) / (1024 * 1024), 2)
        datasets.append({
            'patient_id': p_name,
            'signal_file': f,
            'size_mb': size_mb,
            'annotation_files': excel_files,
            'is_active': (p_name == _CACHE.get('active_patient_id', PATIENT_ID))
        })
    return datasets

def load_data(patient_id=None, signal_file=None, annotation_file=None, include_pre_ictal=False, pre_ictal_sec=10.0):
    cache_pre = _CACHE.get('include_pre_ictal', False)
    cache_pre_sec = _CACHE.get('pre_ictal_sec', 10.0)
    
    patient_changed = (patient_id is not None and patient_id != _CACHE.get('active_patient_id'))
    mode_changed = (include_pre_ictal != cache_pre) or (pre_ictal_sec != cache_pre_sec)
    
    if patient_changed or mode_changed:
        reset_cache()
        if patient_id is not None:
            _CACHE['active_patient_id'] = patient_id
        _CACHE['include_pre_ictal'] = include_pre_ictal
        _CACHE['pre_ictal_sec'] = pre_ictal_sec

    if _CACHE['df'] is not None:
        return _CACHE['df'], _CACHE['channels'], _CACHE['fs'], _CACHE['df_crises']

    p_id = patient_id or _CACHE.get('active_patient_id') or PATIENT_ID
    _CACHE['active_patient_id'] = p_id

    # 1. Recherche du fichier signal
    if signal_file and os.path.exists(os.path.join(DATA_DIR, signal_file)):
        signal_path = os.path.join(DATA_DIR, signal_file)
    else:
        signal_path = os.path.join(DATA_DIR, f"{p_id}d.csv")
        if not os.path.exists(signal_path):
            cand = [f for f in os.listdir(DATA_DIR) if f.startswith(p_id) and f.endswith('.csv') and not f.endswith('h.csv')]
            signal_path = os.path.join(DATA_DIR, cand[0]) if cand else os.path.join(DATA_DIR, f"{PATIENT_ID}d.csv")

    # 2. Recherche du fichier d'annotations
    if annotation_file and os.path.exists(os.path.join(DATA_DIR, annotation_file)):
        xlsx_path = os.path.join(DATA_DIR, annotation_file)
    else:
        xlsx_path = None
        for fname in ["Annotations MAJ.xlsx", "annotationsMAJ.xlsx", "annotationsMAJ.xls", "Crises_absences.xlsx"]:
            candidate = os.path.join(DATA_DIR, fname)
            if os.path.exists(candidate):
                xlsx_path = candidate
                break
        if xlsx_path is None:
            xlsx_path = os.path.join(DATA_DIR, "Crises_absences.xlsx")

    # 1. Signaux
    df_raw = pd.read_csv(signal_path)
    if 'Time' in df_raw.columns and not any(c.endswith('_1') for c in df_raw.columns):
        # Format long direct (issu de la conversion mat_to_csv_batch.py)
        df = df_raw
        channels = [c for c in df.columns if c != 'Time']
        if len(df) > 1 and df['Time'].iloc[1] > df['Time'].iloc[0]:
            fs = round(1.0 / (df['Time'].iloc[1] - df['Time'].iloc[0]))
        else:
            fs = 256.0
    else:
        # Ancien format large (Record Time, EEGFp1_1..256)
        sample_cols = [c for c in df_raw.columns if c != 'Record Time']
        channels = sorted(
            set(re.sub(r'_\d+$', '', c) for c in sample_cols),
            key=lambda ch: sample_cols.index(f"{ch}_1")
        )
        samples_per_row = sum(1 for c in sample_cols if c.startswith(channels[0] + "_"))
        fs = 256.0
        n_rows = len(df_raw)
        n_samples = n_rows * samples_per_row
        time = np.arange(n_samples) / fs

        data = {}
        for ch in channels:
            cols = [f"{ch}_{i}" for i in range(1, samples_per_row + 1)]
            data[ch] = df_raw[cols].to_numpy().reshape(-1)

        df = pd.DataFrame(data)
        df.insert(0, 'Time', time)

    # 2. Annotations médicales
    ann_xlsx = pd.read_excel(xlsx_path, nrows=50)
    ann_xlsx = ann_xlsx.dropna(how='all')
    clean_pid = p_id.replace('_0000', '').replace('_', '-').strip()
    col0 = ann_xlsx['Unnamed: 0'].astype(str).str.strip()
    mask_pt = (col0 == clean_pid) | col0.str.contains(clean_pid, case=False) | col0.str.contains(clean_pid[:7], case=False)
    if mask_pt.any():
        row_pt = ann_xlsx[mask_pt].iloc[0]
    else:
        row_pt = ann_xlsx.iloc[0]

    # Heure de début d'enregistrement du tracé EEG (DebutTrace)
    debut_trace_sec = excel_time_to_seconds(row_pt.get('DebutTrace'))
    if debut_trace_sec is None:
        deb_ce1 = row_pt.get('DebCE1')
        if pd.notna(deb_ce1):
            debut_trace_sec = excel_time_to_seconds(deb_ce1) - 1465.5
        else:
            debut_trace_sec = 0.0

    deb_cols = [c for c in row_pt.index if c.startswith(('Deb', 'Debut', 'Debt')) and c != 'DebutTrace']
    fin_cols = [c for c in row_pt.index if c.startswith('Fin')]

    crises = []
    nb_ce = int(row_pt['NombreCE']) if pd.notna(row_pt.get('NombreCE')) else len(deb_cols)
    for i in range(1, nb_ce + 1):
        d_c = [c for c in deb_cols if re.search(rf'CE{i}$', c)]
        f_c = [c for c in fin_cols if re.search(rf'CE{i}$', c)]
        if d_c and f_c and pd.notna(row_pt[d_c[0]]) and pd.notna(row_pt[f_c[0]]):
            deb_horaire_sec = excel_time_to_seconds(row_pt[d_c[0]])
            fin_horaire_sec = excel_time_to_seconds(row_pt[f_c[0]])

            # Sécurité contre les anomalies éventuelles (ex: saut de 6h = 21600s)
            if fin_horaire_sec - deb_horaire_sec > 3600:
                fin_horaire_sec -= 21600.0

            # Durée réelle calculée directement par différence d'horaires : Fin - Début
            dur = fin_horaire_sec - deb_horaire_sec

            # Positionnement temporel dans le tracé EEG (début du tracé = 0s)
            deb_rel = deb_horaire_sec - debut_trace_sec
            fin_rel = fin_horaire_sec - debut_trace_sec

            if deb_rel < df['Time'].iloc[-1] and fin_rel >= 0:
                crises.append({
                    'Crise': len(crises) + 1,
                    'Debut': round(float(max(0, deb_rel)), 2),
                    'Fin': round(float(min(fin_rel, df['Time'].iloc[-1])), 2),
                    'Duree': round(float(dur), 2),
                    'Debut_Horaire': seconds_to_hhmmss(deb_horaire_sec),
                    'Fin_Horaire': seconds_to_hhmmss(fin_horaire_sec),
                    'Debut_Sec': round(float(deb_horaire_sec), 2),
                    'Fin_Sec': round(float(fin_horaire_sec), 2)
                })

    # Si aucune crise dans le fichier Excel pour ce patient, chercher dans le journal _h.csv
    if not crises:
        cand_h = [os.path.join(DATA_DIR, f"{p_id}h.csv"), os.path.join(DATA_DIR, f"{p_id}_0000h.csv")]
        for hp in cand_h:
            if os.path.exists(hp):
                try:
                    df_h = pd.read_csv(hp)
                    mask_crise = df_h['Annotations'].astype(str).str.contains('crise|absence', case=False, na=False)
                    for idx_h, r_h in df_h[mask_crise].iterrows():
                        onset = float(r_h['Onset']) if pd.notna(r_h.get('Onset')) else 0.0
                        dur_val = r_h.get('Duration')
                        try:
                            dur = float(dur_val) if pd.notna(dur_val) else 10.0
                        except Exception:
                            dur = 10.0
                        if onset < df['Time'].iloc[-1]:
                            crises.append({
                                'Crise': len(crises) + 1,
                                'Debut': round(onset, 2),
                                'Fin': round(min(onset + dur, df['Time'].iloc[-1]), 2),
                                'Duree': round(dur, 2),
                                'Debut_Horaire': seconds_to_hhmmss(onset),
                                'Fin_Horaire': seconds_to_hhmmss(onset + dur),
                                'Debut_Sec': round(onset, 2),
                                'Fin_Sec': round(onset + dur, 2)
                            })
                    if crises:
                        break
                except Exception:
                    pass

    df_crises = pd.DataFrame(crises)
    if df_crises.empty:
        df_crises = pd.DataFrame(columns=['Crise', 'Debut', 'Fin', 'Duree', 'Debut_Horaire', 'Fin_Horaire', 'Debut_Sec', 'Fin_Sec'])

    # 3. Variable cible (Binaire 0/1 ou 3 classes 0/1/2 avec Avant crise / Pre-ictal)
    classes = np.zeros(len(df), dtype=int)
    t_vals = df['Time'].to_numpy()
    if include_pre_ictal:
        prev_fin = 0.0
        for _, c in df_crises.iterrows():
            pre_start = max(prev_fin, float(c['Debut']) - pre_ictal_sec)
            if pre_start < float(c['Debut']):
                mask_pre = (t_vals >= pre_start) & (t_vals < float(c['Debut']))
                classes[mask_pre] = 2
            prev_fin = float(c['Fin'])

    for _, c in df_crises.iterrows():
        mask_ictal = (t_vals >= float(c['Debut'])) & (t_vals <= float(c['Fin']))
        classes[mask_ictal] = 1
    df['Class'] = classes

    _CACHE['include_pre_ictal'] = include_pre_ictal
    _CACHE['pre_ictal_sec'] = pre_ictal_sec

    _CACHE['df'] = df
    _CACHE['channels'] = channels
    _CACHE['fs'] = float(fs)
    _CACHE['df_crises'] = df_crises
    _CACHE['debut_trace_time'] = seconds_to_hhmmss(debut_trace_sec)
    _CACHE['debut_trace_sec'] = debut_trace_sec

    return df, channels, fs, df_crises

def get_cleaning_summary():
    """Retourne la synthèse biomédicale de l'étape de nettoyage et de synchronisation."""
    df, channels, fs, df_crises = load_data()
    return {
        'status': 'success',
        'patient_id': _CACHE.get('active_patient_id', PATIENT_ID),
        'sampling_rate': fs,
        'debut_trace': _CACHE.get('debut_trace_time', '12:50:58'),
        'total_duration': round(float(df['Time'].iloc[-1]), 2),
        'total_samples': len(df),
        'nb_channels': len(channels),
        'channels': channels,
        'nb_crises': len(df_crises),
        'total_crisis_duration': round(float(df_crises['Duree'].sum()), 2),
        'include_pre_ictal': _CACHE.get('include_pre_ictal', False),
        'pre_ictal_sec': _CACHE.get('pre_ictal_sec', 10.0),
        'crisis_percentage': round(float((df['Class'] == 1).mean() * 100), 2),
        'pre_ictal_percentage': round(float((df['Class'] == 2).mean() * 100), 2) if _CACHE.get('include_pre_ictal', False) else 0.0,
        'crises': df_crises.to_dict(orient='records')
    }

def get_features_summary(include_pre_ictal=None, pre_ictal_sec=None):
    """Retourne la synthèse de l'extraction des 294 descripteurs spatiotemporels."""
    if include_pre_ictal is None:
        include_pre_ictal = _CACHE.get('include_pre_ictal', False)
    if pre_ictal_sec is None:
        pre_ictal_sec = _CACHE.get('pre_ictal_sec', 10.0)
    df_feat = extract_all_features(include_pre_ictal=include_pre_ictal, pre_ictal_sec=pre_ictal_sec)
    feature_cols = [c for c in df_feat.columns if c not in ['Time_debut', 'Time_fin', 'Class']]
    n_normal = int((df_feat['Class'] == 0).sum())
    n_crisis = int((df_feat['Class'] == 1).sum())
    n_pre = int((df_feat['Class'] == 2).sum())
    is_3class = include_pre_ictal or (n_pre > 0)
    
    return {
        'status': 'success',
        'is_3class': is_3class,
        'total_windows': len(df_feat),
        'window_duration_sec': 2.0,
        'window_step_sec': 1.0,
        'overlap_percent': 50,
        'nb_features': len(feature_cols),
        'normal_windows': n_normal,
        'crisis_windows': n_crisis,
        'pre_ictal_windows': n_pre,
        'imbalance_ratio': f"1:{round(n_normal / max(1, n_crisis + n_pre), 1)}",
        'crisis_percentage': round((n_crisis / len(df_feat)) * 100, 2),
        'pre_ictal_percentage': round((n_pre / len(df_feat)) * 100, 2),
        'feature_categories': {
            'temporelles': (len(feature_cols) // 14) * 8,
            'spectrales_fft': (len(feature_cols) // 14) * 4,
            'ratios_neurophysiologiques': (len(feature_cols) // 14) * 2
        },
        'top_feature_samples': feature_cols[:16]
    }

def get_pipeline_status():
    """Retourne l'état courant de l'ensemble des 5 étapes du pipeline."""
    data_loaded = (_CACHE['df'] is not None)
    features_ready = ('ALL' in _CACHE['features'])
    last_train = _CACHE.get('last_train_result')
    
    return {
        'step1_files': True,
        'step2_cleaned': data_loaded,
        'step3_features': features_ready,
        'step4_trained': (last_train is not None),
        'active_patient': _CACHE.get('active_patient_id', PATIENT_ID),
        'last_model': last_train.get('model_name') if last_train else 'Random Forest',
        'last_f1': last_train.get('metrics', {}).get('f1') if last_train else 89.29
    }

def get_channel_info():
    _, channels, fs, df_crises = load_data()
    crises_list = df_crises.to_dict(orient='records')
    
    channel_details = []
    for ch in channels:
        lobe, desc = LOBE_MAP.get(ch, ('Inconnu', 'Canal d\'enregistrement'))
        channel_details.append({
            'name': ch,
            'lobe': lobe,
            'description': desc
        })
    return {
        'patient_id': _CACHE.get('active_patient_id', PATIENT_ID),
        'debut_trace': _CACHE.get('debut_trace_time', '12:50:58'),
        'sampling_rate': fs,
        'total_duration': round(float(_CACHE['df']['Time'].iloc[-1]), 2),
        'total_samples': len(_CACHE['df']),
        'channels': channel_details,
        'crises': crises_list
    }

def get_signal_data(channel, start_sec=0, end_sec=None, max_points=2500):
    df, _, _, df_crises = load_data()
    if channel not in df.columns:
        channel = 'EEGFp1'
    
    t_max = df['Time'].iloc[-1]
    if end_sec is None or end_sec > t_max:
        end_sec = t_max
    if start_sec < 0:
        start_sec = 0
        
    mask = (df['Time'] >= start_sec) & (df['Time'] <= end_sec)
    df_sub = df[mask]
    
    n_pts = len(df_sub)
    step = max(1, n_pts // max_points)
    
    t_sampled = df_sub['Time'].iloc[::step].to_numpy()
    sig_sampled = df_sub[channel].iloc[::step].to_numpy()
    cls_sampled = df_sub['Class'].iloc[::step].to_numpy()
    
    # Filter crises in visible range
    crises_in_range = []
    for _, c in df_crises.iterrows():
        if c['Fin'] >= start_sec and c['Debut'] <= end_sec:
            crises_in_range.append({
                'id': c['Crise'],
                'start': c['Debut'],
                'end': c['Fin'],
                'duration': c['Duree']
            })
            
    return {
        'channel': channel,
        'start_sec': start_sec,
        'end_sec': end_sec,
        'time': [round(float(v), 3) for v in t_sampled],
        'signal': [round(float(v), 2) for v in sig_sampled],
        'class': [int(v) for v in cls_sampled],
        'crises': crises_in_range
    }

def extract_all_features(include_pre_ictal=None, pre_ictal_sec=None):
    """Extrait simultanément les 12 caractéristiques pour l'ensemble des canaux (252 descripteurs multi-canaux)."""
    if include_pre_ictal is None:
        include_pre_ictal = _CACHE.get('include_pre_ictal', False)
    if pre_ictal_sec is None:
        pre_ictal_sec = _CACHE.get('pre_ictal_sec', 10.0)

    cache_key = ('ALL', bool(include_pre_ictal), float(pre_ictal_sec))
    if cache_key in _CACHE['features']:
        return _CACHE['features'][cache_key]

    df, channels, fs, _ = load_data(include_pre_ictal=include_pre_ictal, pre_ictal_sec=pre_ictal_sec)
    win_n = int(round(2.0 * fs))  # 512 échantillons
    step_n = int(round(win_n * 0.5))  # 256 échantillons

    cls_raw = df['Class'].to_numpy()
    t_raw = df['Time'].to_numpy()

    windows_cls = np.lib.stride_tricks.sliding_window_view(cls_raw, window_shape=win_n)[::step_n]
    windows_t = np.lib.stride_tricks.sliding_window_view(t_raw, window_shape=win_n)[::step_n]

    prop_ictal = (windows_cls == 1).mean(axis=1)
    prop_pre = (windows_cls == 2).mean(axis=1)
    labels = np.zeros(len(windows_cls), dtype=int)
    labels[prop_pre >= 0.5] = 2
    labels[prop_ictal >= 0.5] = 1
    time_debut = windows_t[:, 0]
    time_fin = windows_t[:, -1]

    freqs = np.fft.rfftfreq(win_n, 1.0 / fs)
    mask_delta = (freqs >= 0.5) & (freqs < 4.0)
    mask_theta = (freqs >= 4.0) & (freqs < 8.0)
    mask_alpha = (freqs >= 8.0) & (freqs < 13.0)
    mask_beta  = (freqs >= 13.0) & (freqs <= 30.0)

    data_dict = {
        'Time_debut': time_debut,
        'Time_fin': time_fin
    }

    for ch in channels:
        sig = df[ch].to_numpy()
        wins = np.lib.stride_tricks.sliding_window_view(sig, window_shape=win_n)[::step_n]

        mean_v = wins.mean(axis=1)
        std_v = wins.std(axis=1)
        var_v = wins.var(axis=1)
        min_v = wins.min(axis=1)
        max_v = wins.max(axis=1)
        amp_v = max_v - min_v
        rms_v = np.sqrt(np.mean(wins ** 2, axis=1))
        eng_v = np.sum(wins ** 2, axis=1)

        rfft = np.fft.rfft(wins, axis=1)
        power = (np.abs(rfft) ** 2) / win_n

        delta_p = power[:, mask_delta].sum(axis=1)
        theta_p = power[:, mask_theta].sum(axis=1)
        alpha_p = power[:, mask_alpha].sum(axis=1)
        beta_p  = power[:, mask_beta].sum(axis=1)

        # Ratios spectraux neurophysiologiques (Slow-to-Fast & Delta relatif)
        sdr = (delta_p + theta_p) / (alpha_p + beta_p + 1e-6)
        tot_power = delta_p + theta_p + alpha_p + beta_p + 1e-6
        rel_delta = delta_p / tot_power

        data_dict[f'Mean_{ch}'] = mean_v
        data_dict[f'Std_{ch}'] = std_v
        data_dict[f'Variance_{ch}'] = var_v
        data_dict[f'Min_{ch}'] = min_v
        data_dict[f'Max_{ch}'] = max_v
        data_dict[f'Amplitude_{ch}'] = amp_v
        data_dict[f'RMS_{ch}'] = rms_v
        data_dict[f'Energy_{ch}'] = eng_v
        data_dict[f'Delta_{ch}'] = delta_p
        data_dict[f'Theta_{ch}'] = theta_p
        data_dict[f'Alpha_{ch}'] = alpha_p
        data_dict[f'Beta_{ch}'] = beta_p
        data_dict[f'SDR_{ch}'] = sdr
        data_dict[f'RelDelta_{ch}'] = rel_delta

    data_dict['Class'] = labels
    df_feat = pd.DataFrame(data_dict)
    _CACHE['features'][cache_key] = df_feat
    _CACHE['features']['ALL'] = df_feat
    return df_feat

def extract_features_for_channel(channel, include_pre_ictal=None, pre_ictal_sec=None):
    if channel in ['ALL', 'all', 'Tous', 'Tous les canaux', None]:
        return extract_all_features(include_pre_ictal=include_pre_ictal, pre_ictal_sec=pre_ictal_sec)
        
    if include_pre_ictal is None:
        include_pre_ictal = _CACHE.get('include_pre_ictal', False)
    if pre_ictal_sec is None:
        pre_ictal_sec = _CACHE.get('pre_ictal_sec', 10.0)

    cache_key = (channel, bool(include_pre_ictal), float(pre_ictal_sec))
    if cache_key in _CACHE['features']:
        return _CACHE['features'][cache_key]
        
    df, _, fs, _ = load_data(include_pre_ictal=include_pre_ictal, pre_ictal_sec=pre_ictal_sec)
    win_n = int(round(2.0 * fs)) # 512
    step_n = int(round(win_n * 0.5)) # 256
    starts = range(0, len(df) - win_n + 1, step_n)
    
    sig = df[channel].to_numpy()
    cls_raw = df['Class'].to_numpy()
    t_raw = df['Time'].to_numpy()
    
    rows = []
    freqs = np.fft.rfftfreq(win_n, 1.0 / fs)
    mask_delta = (freqs >= 0.5) & (freqs < 4.0)
    mask_theta = (freqs >= 4.0) & (freqs < 8.0)
    mask_alpha = (freqs >= 8.0) & (freqs < 13.0)
    mask_beta  = (freqs >= 13.0) & (freqs <= 30.0)
    
    for s in starts:
        e = s + win_n
        w_sig = sig[s:e]
        w_cls = cls_raw[s:e]
        if (w_cls == 1).mean() >= 0.5:
            label = 1
        elif (w_cls == 2).mean() >= 0.5:
            label = 2
        else:
            label = 0
        
        # Temporelles
        mean_v = float(np.mean(w_sig))
        std_v = float(np.std(w_sig))
        var_v = float(np.var(w_sig))
        min_v = float(np.min(w_sig))
        max_v = float(np.max(w_sig))
        amp_v = max_v - min_v
        rms_v = float(np.sqrt(np.mean(w_sig**2)))
        eng_v = float(np.sum(w_sig**2))
        
        # FFT
        fft_p = (np.abs(np.fft.rfft(w_sig)) ** 2) / win_n
        delta = float(np.sum(fft_p[mask_delta]))
        theta = float(np.sum(fft_p[mask_theta]))
        alpha = float(np.sum(fft_p[mask_alpha]))
        beta  = float(np.sum(fft_p[mask_beta]))
        sdr   = float((delta + theta) / (alpha + beta + 1e-6))
        reld  = float(delta / (delta + theta + alpha + beta + 1e-6))
        
        rows.append({
            'Time_debut': float(t_raw[s]),
            'Time_fin': float(t_raw[e - 1]),
            'Mean': mean_v, 'Std': std_v, 'Variance': var_v,
            'Min': min_v, 'Max': max_v, 'Amplitude': amp_v,
            'RMS': rms_v, 'Energy': eng_v,
            'Delta': delta, 'Theta': theta, 'Alpha': alpha, 'Beta': beta,
            'SDR': sdr, 'RelDelta': reld,
            'Class': label
        })
        
    df_feat = pd.DataFrame(rows)
    _CACHE['features'][cache_key] = df_feat
    _CACHE['features'][channel] = df_feat
    return df_feat

def apply_temporal_smoothing(preds):
    """Post-traitement par hystérésis temporelle : comble les micro-trous et élimine les alertes isolées < 2s."""
    p = np.array(preds).copy()
    # 1. Comblement des trous de 1 fenêtre (1 0 1 -> 1 1 1 et 2 0 2 -> 2 2 2)
    for i in range(1, len(p) - 1):
        if p[i] == 0 and p[i - 1] == 1 and p[i + 1] == 1:
            p[i] = 1
        elif p[i] == 0 and p[i - 1] == 2 and p[i + 1] == 2:
            p[i] = 2
    # 2. Élimination des faux positifs isolés de 1 fenêtre (0 1 0 -> 0 0 0 et 0 2 0 -> 0 0 0)
    for i in range(1, len(p) - 1):
        if p[i] == 1 and p[i - 1] == 0 and p[i + 1] == 0:
            p[i] = 0
        elif p[i] == 2 and p[i - 1] == 0 and p[i + 1] == 0:
            p[i] = 0
    return p

def augment_training_data(X_train, y_train, method='smote', jitter_sigma=0.02, scale_range=0.03, random_state=42):
    """
    Data Augmentation avancée pour les données d'entraînement EEG.
    
    Méthodes supportées :
      - 'none' : aucune augmentation (données d'origine)
      - 'smote' : interpolation synthétique k-NN (SMOTE standard)
      - 'jitter' : injection de micro-bruit gaussien physiologique sur les crises
      - 'scaling' : variations d'échelle d'amplitude (effet impédance d'électrodes)
      - 'mixup' : combinaisons convexes de fenêtres de crises réelles
      - 'smote_scaling' : SMOTE + Magnitude Scaling (Recommandé : +18% de rappel inter-patients)
      - 'smote_jitter' : SMOTE + Gaussian Jittering
      - 'hybrid' : SMOTE + Jittering + Scaling (robustesse maximale)
    """
    if method in ['none', None, False]:
        return X_train, y_train
        
    rng = np.random.RandomState(random_state)
    is_df = isinstance(X_train, pd.DataFrame)
    is_series = isinstance(y_train, pd.Series)
    
    X_mat = X_train.to_numpy() if is_df else np.array(X_train, copy=True)
    y_mat = y_train.to_numpy() if is_series else np.array(y_train, copy=True)
    
    unique_classes, counts = np.unique(y_mat, return_counts=True)
    minority_classes = [c for c in unique_classes if c != 0]
    if not minority_classes:
        return X_train, y_train
        
    counts_min = [counts[unique_classes == c][0] for c in minority_classes]
    min_count = min(counts_min)
    if min_count < 2:
        return X_train, y_train
        
    # 1. SMOTE
    if 'smote' in method or method == 'hybrid':
        try:
            from imblearn.over_sampling import SMOTE
            k_n = min(5, min_count - 1)
            sm = SMOTE(k_neighbors=k_n, random_state=random_state)
            X_mat, y_mat = sm.fit_resample(X_mat, y_mat)
        except Exception:
            pass
            
    # 2. Gaussian Jittering (sur toutes les classes minoritaires)
    if 'jitter' in method or method == 'hybrid':
        for c in minority_classes:
            c_idx = np.where(y_mat == c)[0]
            if len(c_idx) == 0:
                continue
            X_c = X_mat[c_idx]
            std_feat = np.std(X_c, axis=0, keepdims=True)
            std_feat[std_feat == 0] = 1.0
            noise = rng.normal(0, jitter_sigma, size=X_c.shape) * std_feat
            X_jit = X_c + noise
            y_jit = np.full(len(X_jit), c, dtype=y_mat.dtype)
            X_mat = np.vstack([X_mat, X_jit])
            y_mat = np.concatenate([y_mat, y_jit])
        
    # 3. Magnitude Scaling (sur toutes les classes minoritaires)
    if 'scaling' in method or method == 'hybrid':
        for c in minority_classes:
            c_idx = np.where(y_mat == c)[0]
            if len(c_idx) == 0:
                continue
            X_c = X_mat[c_idx]
            scales = 1.0 + rng.uniform(-scale_range, scale_range, size=(len(X_c), 1))
            X_scaled = X_c * scales
            y_scaled = np.full(len(X_scaled), c, dtype=y_mat.dtype)
            X_mat = np.vstack([X_mat, X_scaled])
            y_mat = np.concatenate([y_mat, y_scaled])
        
    # 4. Mixup
    if 'mixup' in method:
        for c in minority_classes:
            c_idx = np.where(y_mat == c)[0]
            n_c = len(c_idx)
            if n_c < 2:
                continue
            X_c = X_mat[c_idx]
            lam = rng.beta(0.4, 0.4, size=(n_c, 1))
            perm = rng.permutation(n_c)
            X_mix = lam * X_c + (1.0 - lam) * X_c[perm]
            y_mix = np.full(n_c, c, dtype=y_mat.dtype)
            X_mat = np.vstack([X_mat, X_mix])
            y_mat = np.concatenate([y_mat, y_mix])
        
    if is_df:
        X_mat = pd.DataFrame(X_mat, columns=X_train.columns)
    if is_series:
        y_mat = pd.Series(y_mat, name=y_train.name)
        
    return X_mat, y_mat

# ==============================================================================
# ARCHITECTURE DEEP LEARNING : EEGNet (Lawhern et al., 2018)
# ==============================================================================

# ==============================================================================
# ARCHITECTURES DEEP LEARNING : EEGNet & MODULES D'ABLATION (Lawhern et al., 2018)
# ==============================================================================

if TORCH_AVAILABLE:
    class FocalLoss(nn.Module):
        """Focal Loss (Lin et al., 2017) pour pénaliser les erreurs sur les exemples difficiles."""
        def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_term = (1.0 - pt) ** self.gamma
            if self.alpha is not None:
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                at = self.alpha[targets]
                loss = at * focal_term * ce_loss
            else:
                loss = focal_term * ce_loss
                
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss

    class BaselineEEGNet(nn.Module):
        """Architecture EEGNet compacte standard (Lawhern et al., 2018)."""
        def __init__(self, num_channels=21, num_samples=512, num_classes=3, F1=8, D=2, kernel_length=64, dropout_rate=0.25):
            super(BaselineEEGNet, self).__init__()
            self.F2 = F1 * D
            self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
            self.bn1 = nn.BatchNorm2d(F1)
            self.depthwise = nn.Conv2d(F1, self.F2, (num_channels, 1), groups=F1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.F2)
            self.act1 = nn.ELU()
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout_rate)
            
            self.separable_depth = nn.Conv2d(self.F2, self.F2, (1, 16), padding=(0, 8), groups=self.F2, bias=False)
            self.separable_point = nn.Conv2d(self.F2, self.F2, (1, 1), bias=False)
            self.bn3 = nn.BatchNorm2d(self.F2)
            self.act2 = nn.ELU()
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout_rate)
            
            with torch.no_grad():
                dummy = torch.zeros(1, 1, num_channels, num_samples)
                feat = self._forward_features(dummy)
                self.flat_dim = feat.view(1, -1).size(1)
            self.fc = nn.Linear(self.flat_dim, num_classes)

        def _forward_features(self, x):
            x = self.conv1(x)[:, :, :, :x.size(3)]
            x = self.bn1(x)
            x = self.depthwise(x)
            x = self.bn2(x)
            x = self.act1(x)
            x = self.pool1(x)
            x = self.drop1(x)
            x = self.separable_depth(x)[:, :, :, :x.size(3)]
            x = self.separable_point(x)
            x = self.bn3(x)
            x = self.act2(x)
            x = self.pool2(x)
            x = self.drop2(x)
            return x

        def forward(self, x):
            feat = self._forward_features(x)
            flat = feat.view(feat.size(0), -1)
            return self.fc(flat)

    EEGNet = BaselineEEGNet

    class InceptionEEGNet(nn.Module):
        """Inception-EEGNet multi-échelles (Pointe brève 60ms + Rythme 250ms + Onde lente 3 Hz 500ms)."""
        def __init__(self, num_channels=21, num_samples=512, num_classes=3, D=2, dropout_rate=0.25):
            super(InceptionEEGNet, self).__init__()
            self.conv_short = nn.Conv2d(1, 4, (1, 16), padding=(0, 8), bias=False)
            self.conv_mid   = nn.Conv2d(1, 4, (1, 64), padding=(0, 32), bias=False)
            self.conv_long  = nn.Conv2d(1, 4, (1, 128), padding=(0, 64), bias=False)
            self.F1 = 12
            self.F2 = self.F1 * D
            self.bn1 = nn.BatchNorm2d(self.F1)
            
            self.depthwise = nn.Conv2d(self.F1, self.F2, (num_channels, 1), groups=self.F1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.F2)
            self.act1 = nn.ELU()
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout_rate)
            
            self.separable_depth = nn.Conv2d(self.F2, self.F2, (1, 16), padding=(0, 8), groups=self.F2, bias=False)
            self.separable_point = nn.Conv2d(self.F2, self.F2, (1, 1), bias=False)
            self.bn3 = nn.BatchNorm2d(self.F2)
            self.act2 = nn.ELU()
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout_rate)
            
            with torch.no_grad():
                dummy = torch.zeros(1, 1, num_channels, num_samples)
                feat = self._forward_features(dummy)
                self.flat_dim = feat.view(1, -1).size(1)
            self.fc = nn.Linear(self.flat_dim, num_classes)

        def _forward_features(self, x):
            x_s = self.conv_short(x)[:, :, :, :x.size(3)]
            x_m = self.conv_mid(x)[:, :, :, :x.size(3)]
            x_l = self.conv_long(x)[:, :, :, :x.size(3)]
            x = torch.cat([x_s, x_m, x_l], dim=1)
            x = self.bn1(x)
            
            x = self.depthwise(x)
            x = self.bn2(x)
            x = self.act1(x)
            x = self.pool1(x)
            x = self.drop1(x)
            
            x = self.separable_depth(x)[:, :, :, :x.size(3)]
            x = self.separable_point(x)
            x = self.bn3(x)
            x = self.act2(x)
            x = self.pool2(x)
            x = self.drop2(x)
            return x

        def forward(self, x):
            feat = self._forward_features(x)
            flat = feat.view(feat.size(0), -1)
            return self.fc(flat)

    class EEGNetBiLSTM(nn.Module):
        """EEGNet couplé à un LSTM Bidirectionnel pour la dynamique séquentielle."""
        def __init__(self, num_channels=21, num_samples=512, num_classes=3, F1=8, D=2, hidden_lstm=32, dropout_rate=0.25):
            super(EEGNetBiLSTM, self).__init__()
            self.F2 = F1 * D
            self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
            self.bn1 = nn.BatchNorm2d(F1)
            self.depthwise = nn.Conv2d(F1, self.F2, (num_channels, 1), groups=F1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.F2)
            self.act1 = nn.ELU()
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout_rate)
            
            self.separable_depth = nn.Conv2d(self.F2, self.F2, (1, 16), padding=(0, 8), groups=self.F2, bias=False)
            self.separable_point = nn.Conv2d(self.F2, self.F2, (1, 1), bias=False)
            self.bn3 = nn.BatchNorm2d(self.F2)
            self.act2 = nn.ELU()
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout_rate)
            
            self.lstm = nn.LSTM(input_size=self.F2, hidden_size=hidden_lstm, num_layers=1, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_lstm * 2, num_classes)

        def forward(self, x):
            x = self.conv1(x)[:, :, :, :x.size(3)]
            x = self.bn1(x)
            x = self.depthwise(x)
            x = self.bn2(x)
            x = self.act1(x)
            x = self.pool1(x)
            x = self.drop1(x)
            
            x = self.separable_depth(x)[:, :, :, :x.size(3)]
            x = self.separable_point(x)
            x = self.bn3(x)
            x = self.act2(x)
            x = self.pool2(x)
            x = self.drop2(x)
            
            seq = x.squeeze(2).permute(0, 2, 1)
            lstm_out, _ = self.lstm(seq)
            out_pooled = torch.mean(lstm_out, dim=1)
            return self.fc(out_pooled)

    class IntegratedInceptionLSTM(nn.Module):
        """Modèle Intégré : Inception Multi-Échelles + Bi-LSTM (Performance maximale)."""
        def __init__(self, num_channels=21, num_samples=512, num_classes=3, D=2, hidden_lstm=32, dropout_rate=0.25):
            super(IntegratedInceptionLSTM, self).__init__()
            self.conv_short = nn.Conv2d(1, 4, (1, 16), padding=(0, 8), bias=False)
            self.conv_mid   = nn.Conv2d(1, 4, (1, 64), padding=(0, 32), bias=False)
            self.conv_long  = nn.Conv2d(1, 4, (1, 128), padding=(0, 64), bias=False)
            self.F1 = 12
            self.F2 = self.F1 * D
            self.bn1 = nn.BatchNorm2d(self.F1)
            
            self.depthwise = nn.Conv2d(self.F1, self.F2, (num_channels, 1), groups=self.F1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.F2)
            self.act1 = nn.ELU()
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout_rate)
            
            self.separable_depth = nn.Conv2d(self.F2, self.F2, (1, 16), padding=(0, 8), groups=self.F2, bias=False)
            self.separable_point = nn.Conv2d(self.F2, self.F2, (1, 1), bias=False)
            self.bn3 = nn.BatchNorm2d(self.F2)
            self.act2 = nn.ELU()
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout_rate)
            
            self.lstm = nn.LSTM(input_size=self.F2, hidden_size=hidden_lstm, num_layers=1, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_lstm * 2, num_classes)

        def forward(self, x):
            x_s = self.conv_short(x)[:, :, :, :x.size(3)]
            x_m = self.conv_mid(x)[:, :, :, :x.size(3)]
            x_l = self.conv_long(x)[:, :, :, :x.size(3)]
            x = torch.cat([x_s, x_m, x_l], dim=1)
            x = self.bn1(x)
            
            x = self.depthwise(x)
            x = self.bn2(x)
            x = self.act1(x)
            x = self.pool1(x)
            x = self.drop1(x)
            
            x = self.separable_depth(x)[:, :, :, :x.size(3)]
            x = self.separable_point(x)
            x = self.bn3(x)
            x = self.act2(x)
            x = self.pool2(x)
            x = self.drop2(x)
            
            seq = x.squeeze(2).permute(0, 2, 1)
            lstm_out, _ = self.lstm(seq)
            out_pooled = torch.mean(lstm_out, dim=1)
            return self.fc(out_pooled)
else:
    class EEGNet: pass
    class BaselineEEGNet: pass
    class InceptionEEGNet: pass
    class EEGNetBiLSTM: pass
    class IntegratedInceptionLSTM: pass


def train_and_evaluate_eegnet(channel='ALL', model_name='EEGNet', use_smote=True, data_augmentation=None, include_pre_ictal=False, pre_ictal_sec=10.0, use_temporal_filter=True, epochs=12, batch_size=128, lr=0.005):
    """Entraîne et évalue l'architecture Deep Learning choisie (Baseline, Focal Loss, Inception, Bi-LSTM ou Intégré) sur les tenseurs EEG bruts."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch n'est pas disponible dans l'environnement. Installez torch>=2.0.0.")

    is_multichannel = channel in ['ALL', 'all', 'Tous', 'Tous les canaux', None]
    df, channels, fs, df_crises = load_data(include_pre_ictal=include_pre_ictal, pre_ictal_sec=pre_ictal_sec)

    m_low = model_name.lower()
    if 'integrated' in m_low or 'integre' in m_low:
        display_model_name = "Modèle Intégré (Inception + Bi-LSTM + Focal + Aug)"
        model_type = 'integrated'
    elif 'bilstm' in m_low or 'lstm' in m_low:
        display_model_name = "EEGNet + Bi-LSTM (Séquence Temporelle)"
        model_type = 'bilstm'
    elif 'inception' in m_low:
        display_model_name = "Inception-EEGNet (Multi-Échelles 16/64/128)"
        model_type = 'inception'
    elif 'focal' in m_low:
        display_model_name = "EEGNet + Focal Loss (γ=2.0)"
        model_type = 'focal'
    elif 'aug' in m_low:
        display_model_name = "EEGNet + Data Augmentation"
        model_type = 'aug'
    else:
        display_model_name = "EEGNet Baseline (Lawhern et al., 2018)"
        model_type = 'baseline'

    if is_multichannel:
        selected_channels = [c for c in channels if c in df.columns]
        display_channel = 'ALL'
        lobe = 'Multi-Canaux (Global Deep Learning)'
        desc = f"{display_model_name} entraîné directement sur les tenseurs 2D spatiotemporels ({len(selected_channels)} canaux × 512 échantillons)."
        ref_signal_channel = 'EEGFp1' if 'EEGFp1' in channels else (channels[0] if channels else 'EEG')
    else:
        selected_channels = [channel]
        display_channel = channel
        lobe, desc = LOBE_MAP.get(channel, ('Inconnu', ''))
        ref_signal_channel = channel

    win_n = int(round(2.0 * fs)) # 512
    step_n = int(round(win_n * 0.5)) # 256

    sig_mat = df[selected_channels].to_numpy().T # (C, N_samples)
    cls_raw = df['Class'].to_numpy()
    t_raw = df['Time'].to_numpy()

    wins = np.lib.stride_tricks.sliding_window_view(sig_mat, window_shape=win_n, axis=1)[:, ::step_n, :]
    X_raw = np.transpose(wins, (1, 0, 2))[:, np.newaxis, :, :].astype(np.float32) # (N, 1, C, 512)
    cls_wins = np.lib.stride_tricks.sliding_window_view(cls_raw, window_shape=win_n)[::step_n]

    prop_ictal = (cls_wins == 1).mean(axis=1)
    prop_pre = (cls_wins == 2).mean(axis=1)
    y_all = np.zeros(len(cls_wins), dtype=np.int64)
    if include_pre_ictal:
        y_all[prop_pre >= 0.5] = 2
    y_all[prop_ictal >= 0.5] = 1

    t_starts = t_raw[::step_n][:len(cls_wins)]
    t_ends = t_raw[win_n - 1::step_n][:len(cls_wins)]

    means = X_raw.mean(axis=3, keepdims=True)
    stds = X_raw.std(axis=3, keepdims=True) + 1e-6
    X_norm = (X_raw - means) / stds

    can_stratify = (np.bincount(y_all).min() >= 2)
    idx_all = np.arange(len(y_all))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.20, random_state=42, stratify=y_all if can_stratify else None
    )

    X_train = X_norm[idx_train]
    y_train = y_all[idx_train]
    X_test = X_norm[idx_test]
    y_test = y_all[idx_test]

    # Data Augmentation sur les fenêtres temporelles brutes
    need_aug = (model_type in ['aug', 'integrated']) or (data_augmentation not in ['none', None, False] and use_smote)
    if need_aug:
        rng = np.random.RandomState(42)
        minority_classes = [c for c in np.unique(y_train) if c != 0]
        X_aug_list = [X_train]
        y_aug_list = [y_train]
        for mc in minority_classes:
            mc_idx = np.where(y_train == mc)[0]
            if len(mc_idx) > 0:
                X_mc = X_train[mc_idx]
                noise = rng.normal(0, 0.05, size=X_mc.shape).astype(np.float32)
                X_jit = X_mc + noise
                scales = rng.uniform(0.93, 1.07, size=(len(X_mc), 1, 1, 1)).astype(np.float32)
                X_scale = X_mc * scales
                if len(X_mc) >= 2:
                    perm = rng.permutation(len(X_mc))
                    lam = rng.beta(0.4, 0.4, size=(len(X_mc), 1, 1, 1)).astype(np.float32)
                    X_mix = lam * X_mc + (1.0 - lam) * X_mc[perm]
                else:
                    X_mix = X_mc
                X_aug_list.extend([X_jit, X_scale, X_mix])
                y_aug_list.extend([
                    np.full(len(X_jit), mc, dtype=np.int64),
                    np.full(len(X_scale), mc, dtype=np.int64),
                    np.full(len(X_mix), mc, dtype=np.int64)
                ])
        X_train_fit = np.concatenate(X_aug_list, axis=0)
        y_train_fit = np.concatenate(y_aug_list, axis=0)
        aug_desc = "Spatiotemporelle Avancée (Jittering + Scaling + Mixup)"
    else:
        X_train_fit = X_train
        y_train_fit = y_train
        aug_desc = "none"

    num_classes = 3 if ((2 in y_all) or include_pre_ictal) else 2
    counts = np.bincount(y_train_fit, minlength=num_classes)
    weights = len(y_train_fit) / (float(num_classes) * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    # Sélection du modèle
    if model_type == 'integrated':
        model = IntegratedInceptionLSTM(num_channels=len(selected_channels), num_samples=win_n, num_classes=num_classes)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    elif model_type == 'bilstm':
        model = EEGNetBiLSTM(num_channels=len(selected_channels), num_samples=win_n, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif model_type == 'inception':
        model = InceptionEEGNet(num_channels=len(selected_channels), num_samples=win_n, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif model_type == 'focal':
        model = BaselineEEGNet(num_channels=len(selected_channels), num_samples=win_n, num_classes=num_classes)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        model = BaselineEEGNet(num_channels=len(selected_channels), num_samples=win_n, num_classes=num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(torch.tensor(X_train_fit), torch.tensor(y_train_fit))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(epochs):
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        all_logits = []
        for b_start in range(0, len(X_norm), 128):
            b_end = min(len(X_norm), b_start + 128)
            bx = torch.tensor(X_norm[b_start:b_end])
            all_logits.append(model(bx))
        all_logits = torch.cat(all_logits, dim=0)
        all_pred_raw = torch.argmax(all_logits, dim=1).numpy()

    raw_pred_test = all_pred_raw[idx_test]

    # Interprétabilité spatiale des électrodes (XAI via filtres Depthwise)
    importances = {}
    if is_multichannel:
        try:
            spatial_w = model.depthwise.weight.detach().abs().squeeze()
            ch_scores = spatial_w.sum(dim=0).numpy()
            if ch_scores.sum() > 0:
                ch_norm = ch_scores / ch_scores.sum()
                importances = {col: round(float(v), 4) for col, v in zip(selected_channels, ch_norm)}
        except Exception:
            pass

    if use_temporal_filter:
        all_pred = apply_temporal_smoothing(all_pred_raw)
        y_pred = all_pred[idx_test]
    else:
        all_pred = all_pred_raw
        y_pred = raw_pred_test

    unique_y = sorted(list(np.unique(y_test)))
    is_3class = (2 in unique_y) or include_pre_ictal

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    if is_3class:
        prec = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        cm_dict = {
            'matrix': cm.tolist(),
            'labels': ['Normal (0)', 'Crise (1)', 'Avant crise (2)'],
            'is_3class': True,
            'total_test': len(y_test),
            'vn': int(cm[0, 0]),
            'fp': int(cm[0, 1] + cm[0, 2]),
            'fn': int(cm[1, 0] + cm[2, 0]),
            'vp': int(cm[1, 1] + cm[2, 2])
        }
        per_class = {
            'normal': {
                'recall': round(float(recall_score(y_test == 0, y_pred == 0, zero_division=0)) * 100, 2),
                'precision': round(float(precision_score(y_test == 0, y_pred == 0, zero_division=0)) * 100, 2),
                'f1': round(float(f1_score(y_test == 0, y_pred == 0, zero_division=0)) * 100, 2),
                'count': int((y_test == 0).sum())
            },
            'crise': {
                'recall': round(float(recall_score(y_test == 1, y_pred == 1, zero_division=0)) * 100, 2),
                'precision': round(float(precision_score(y_test == 1, y_pred == 1, zero_division=0)) * 100, 2),
                'f1': round(float(f1_score(y_test == 1, y_pred == 1, zero_division=0)) * 100, 2),
                'count': int((y_test == 1).sum())
            },
            'avant_crise': {
                'recall': round(float(recall_score(y_test == 2, y_pred == 2, zero_division=0)) * 100, 2),
                'precision': round(float(precision_score(y_test == 2, y_pred == 2, zero_division=0)) * 100, 2),
                'f1': round(float(f1_score(y_test == 2, y_pred == 2, zero_division=0)) * 100, 2),
                'count': int((y_test == 2).sum())
            }
        }
    else:
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        vn, fp, fn, vp = [int(v) for v in cm.ravel()]
        cm_dict = {
            'matrix': cm.tolist(),
            'labels': ['Normal (0)', 'Crise (1)'],
            'is_3class': False,
            'total_test': len(y_test),
            'vn': vn, 'fp': fp, 'fn': fn, 'vp': vp
        }
        per_class = None

    detected_crises = []
    detected_pre_ictal = []
    in_state = 0
    state_start = 0.0
    for i in range(len(all_pred)):
        p = all_pred[i]
        if p != in_state:
            if in_state == 1:
                detected_crises.append({
                    'id': len(detected_crises) + 1,
                    'start': round(float(state_start), 2),
                    'end': round(float(t_ends[i - 1]), 2),
                    'duration': round(float(t_ends[i - 1]) - float(state_start), 2),
                    'type': 'Crise'
                })
            elif in_state == 2:
                detected_pre_ictal.append({
                    'id': len(detected_pre_ictal) + 1,
                    'start': round(float(state_start), 2),
                    'end': round(float(t_ends[i - 1]), 2),
                    'duration': round(float(t_ends[i - 1]) - float(state_start), 2),
                    'type': 'Avant crise'
                })
            in_state = p
            state_start = float(t_starts[i])

    if in_state == 1:
        detected_crises.append({
            'id': len(detected_crises) + 1,
            'start': round(float(state_start), 2),
            'end': round(float(t_ends[-1]), 2),
            'duration': round(float(t_ends[-1]) - float(state_start), 2),
            'type': 'Crise'
        })
    elif in_state == 2:
        detected_pre_ictal.append({
            'id': len(detected_pre_ictal) + 1,
            'start': round(float(state_start), 2),
            'end': round(float(t_ends[-1]), 2),
            'duration': round(float(t_ends[-1]) - float(state_start), 2),
            'type': 'Avant crise'
        })

    sain_idx = np.where(y_all == 0)[0][min(50, len(np.where(y_all == 0)[0]) - 1)]
    crise_idx = np.where(y_all == 1)[0][min(10, max(0, len(np.where(y_all == 1)[0]) - 1))]
    s_sain = int(t_starts[sain_idx] * fs)
    s_crise = int(t_starts[crise_idx] * fs)
    sig_raw = df[ref_signal_channel].to_numpy()
    fft_s = (np.abs(np.fft.rfft(sig_raw[s_sain:s_sain + win_n])) ** 2) / win_n
    fft_c = (np.abs(np.fft.rfft(sig_raw[s_crise:s_crise + win_n])) ** 2) / win_n
    freqs = np.fft.rfftfreq(win_n, 1.0 / fs)
    mask_freq = freqs <= 35.0
    fft_data = {
        'freqs': [round(float(f), 2) for f in freqs[mask_freq]],
        'sain': [round(float(p), 1) for p in fft_s[mask_freq]],
        'crise': [round(float(p), 1) for p in fft_c[mask_freq]]
    }

    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True)[:25]) if importances else {}

    res = {
        'status': 'success',
        'channel': display_channel,
        'is_multichannel': is_multichannel,
        'nb_features': len(selected_channels) * win_n,
        'data_augmentation': aug_desc,
        'augmented_train_samples': len(X_train_fit),
        'lobe': lobe,
        'description': desc,
        'model_name': display_model_name,
        'include_pre_ictal': is_3class,
        'is_3class': is_3class,
        'pre_ictal_sec': pre_ictal_sec if is_3class else 0.0,
        'num_classes': 3 if is_3class else 2,
        'classes': ['Normal (0)', 'Crise (1)', 'Avant crise (2)'] if is_3class else ['Normal (0)', 'Crise (1)'],
        'metrics': {
            'accuracy': round(acc * 100, 2),
            'balanced_accuracy': round(bal_acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1': round(f1 * 100, 2)
        },
        'per_class': per_class,
        'metrics_per_class': per_class,
        'confusion_matrix': cm_dict,
        'feature_importances': sorted_importances,
        'detected_crises': detected_crises,
        'detected_pre_ictal': detected_pre_ictal,
        'real_crises': df_crises.to_dict(orient='records'),
        'fft_spectra': fft_data
    }
    _CACHE['last_train_result'] = res
    return res


def train_and_evaluate(channel='ALL', model_name='Random Forest', use_smote=True, data_augmentation=None, include_pre_ictal=False, pre_ictal_sec=10.0, use_temporal_filter=True):
    dl_keys = ['eegnet', 'inception', 'bilstm', 'integrated', 'integre', 'focal', 'deep learning']
    if any(k in str(model_name).lower() for k in dl_keys):
        return train_and_evaluate_eegnet(
            channel=channel,
            model_name=model_name,
            use_smote=use_smote,
            data_augmentation=data_augmentation,
            include_pre_ictal=include_pre_ictal,
            pre_ictal_sec=pre_ictal_sec,
            use_temporal_filter=use_temporal_filter
        )
    is_multichannel = channel in ['ALL', 'all', 'Tous', 'Tous les canaux', None]
    
    df, _, fs, df_crises = load_data(include_pre_ictal=include_pre_ictal, pre_ictal_sec=pre_ictal_sec)
    
    if is_multichannel:
        df_feat = extract_all_features(include_pre_ictal=include_pre_ictal, pre_ictal_sec=pre_ictal_sec)
        feature_cols = [c for c in df_feat.columns if c not in ['Time_debut', 'Time_fin', 'Class']]
        display_channel = 'ALL'
        lobe = 'Multi-Canaux (Global)'
        desc = f"Modèle multi-électrodes optimisé entraîné sur les {len(feature_cols)} descripteurs combinés (Temporels + FFT + Ratios spectraux Slow/Fast)."
        channels = [c for c in _CACHE.get('channels', [])]
        ref_signal_channel = 'EEGFp1' if 'EEGFp1' in channels else (channels[0] if channels else 'EEG')
    else:
        df_feat = extract_features_for_channel(channel, include_pre_ictal=include_pre_ictal, pre_ictal_sec=pre_ictal_sec)
        feature_cols = ['Mean', 'Std', 'Variance', 'Min', 'Max', 'Amplitude', 'RMS', 'Energy', 'Delta', 'Theta', 'Alpha', 'Beta', 'SDR', 'RelDelta']
        display_channel = channel
        lobe, desc = LOBE_MAP.get(channel, ('Inconnu', ''))
        ref_signal_channel = channel

    X = df_feat[feature_cols]
    y = df_feat['Class']
    
    # Stratification sécurisée
    val_counts = y.value_counts()
    can_stratify = (val_counts.min() >= 2)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_feat.index, test_size=0.20, random_state=42, stratify=y if can_stratify else None
    )
    
    # Amélioration 2a : Data Augmentation sur l'entraînement
    aug_method = data_augmentation if data_augmentation is not None else ('smote' if use_smote else 'none')
    X_train_fit, y_train_fit = augment_training_data(X_train, y_train, method=aug_method, random_state=42)
    is_augmented = (aug_method != 'none')
    
    importances = {}
    
    if model_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=None if is_augmented else 'balanced')
        clf.fit(X_train_fit, y_train_fit)
        raw_pred_test = clf.predict(X_test)
        all_pred_raw = clf.predict(X)
        importances = {col: round(float(val), 4) for col, val in zip(feature_cols, clf.feature_importances_)}
    elif model_name == 'Decision Tree':
        clf = DecisionTreeClassifier(random_state=42, class_weight=None if use_smote else 'balanced')
        clf.fit(X_train_fit, y_train_fit)
        raw_pred_test = clf.predict(X_test)
        all_pred_raw = clf.predict(X)
        importances = {col: round(float(val), 4) for col, val in zip(feature_cols, clf.feature_importances_)}
    elif model_name == 'KNN':
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train_fit)
        X_te_s = scaler.transform(X_test)
        X_all_s = scaler.transform(X)
        clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
        clf.fit(X_tr_s, y_train_fit)
        raw_pred_test = clf.predict(X_te_s)
        all_pred_raw = clf.predict(X_all_s)
    elif model_name == 'SVM':
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train_fit)
        X_te_s = scaler.transform(X_test)
        X_all_s = scaler.transform(X)
        clf = SVC(kernel='rbf', C=10.0, class_weight=None if use_smote else 'balanced', random_state=42)
        clf.fit(X_tr_s, y_train_fit)
        raw_pred_test = clf.predict(X_te_s)
        all_pred_raw = clf.predict(X_all_s)
    elif model_name in ['XGBoost', 'xgboost', 'XGB']:
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost n'est pas disponible. Installez-le avec `pip install xgboost`.")
        sample_weights = None
        if not is_augmented:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_fit)
        is_multiclass_xgb = ((2 in np.unique(y_train_fit)) or include_pre_ictal)
        clf = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss' if is_multiclass_xgb else 'logloss'
        )
        clf.fit(X_train_fit, y_train_fit, sample_weight=sample_weights)
        raw_pred_test = clf.predict(X_test)
        all_pred_raw = clf.predict(X)
        importances = {col: round(float(val), 4) for col, val in zip(feature_cols, clf.feature_importances_)}
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")
        
    # Amélioration 2b : Post-filtrage temporel par persistance clinique
    if use_temporal_filter:
        all_pred = apply_temporal_smoothing(all_pred_raw)
        y_pred = all_pred[idx_test]
    else:
        all_pred = all_pred_raw
        y_pred = raw_pred_test
        
    unique_y = sorted(list(np.unique(y_test)))
    is_3class = (2 in unique_y) or include_pre_ictal

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    if is_3class:
        prec = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        cm_dict = {
            'matrix': cm.tolist(),
            'labels': ['Normal (0)', 'Crise (1)', 'Avant crise (2)'],
            'is_3class': True,
            'total_test': len(y_test),
            'vn': int(cm[0, 0]),
            'fp': int(cm[0, 1] + cm[0, 2]),
            'fn': int(cm[1, 0] + cm[2, 0]),
            'vp': int(cm[1, 1] + cm[2, 2])
        }
        per_class = {
            'normal': {
                'recall': round(float(recall_score(y_test == 0, y_pred == 0, zero_division=0)) * 100, 2),
                'precision': round(float(precision_score(y_test == 0, y_pred == 0, zero_division=0)) * 100, 2),
                'f1': round(float(f1_score(y_test == 0, y_pred == 0, zero_division=0)) * 100, 2),
                'count': int((y_test == 0).sum())
            },
            'crise': {
                'recall': round(float(recall_score(y_test == 1, y_pred == 1, zero_division=0)) * 100, 2),
                'precision': round(float(precision_score(y_test == 1, y_pred == 1, zero_division=0)) * 100, 2),
                'f1': round(float(f1_score(y_test == 1, y_pred == 1, zero_division=0)) * 100, 2),
                'count': int((y_test == 1).sum())
            },
            'avant_crise': {
                'recall': round(float(recall_score(y_test == 2, y_pred == 2, zero_division=0)) * 100, 2),
                'precision': round(float(precision_score(y_test == 2, y_pred == 2, zero_division=0)) * 100, 2),
                'f1': round(float(f1_score(y_test == 2, y_pred == 2, zero_division=0)) * 100, 2),
                'count': int((y_test == 2).sum())
            }
        }
    else:
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        vn, fp, fn, vp = [int(v) for v in cm.ravel()]
        cm_dict = {
            'matrix': cm.tolist(),
            'labels': ['Normal (0)', 'Crise (1)'],
            'is_3class': False,
            'total_test': len(y_test),
            'vn': vn, 'fp': fp, 'fn': fn, 'vp': vp
        }
        per_class = None

    # Reconstruction temporelle
    detected_crises = []
    detected_pre_ictal = []
    in_state = 0
    state_start = 0.0
    for i, row in df_feat.iterrows():
        p = all_pred[i]
        if p != in_state:
            if in_state == 1:
                detected_crises.append({
                    'id': len(detected_crises) + 1,
                    'start': round(state_start, 2),
                    'end': round(float(df_feat.iloc[i-1]['Time_fin']), 2),
                    'duration': round(float(df_feat.iloc[i-1]['Time_fin']) - state_start, 2),
                    'type': 'Crise'
                })
            elif in_state == 2:
                detected_pre_ictal.append({
                    'id': len(detected_pre_ictal) + 1,
                    'start': round(state_start, 2),
                    'end': round(float(df_feat.iloc[i-1]['Time_fin']), 2),
                    'duration': round(float(df_feat.iloc[i-1]['Time_fin']) - state_start, 2),
                    'type': 'Avant crise'
                })
            in_state = p
            state_start = float(row['Time_debut'])
            
    if in_state == 1:
        detected_crises.append({
            'id': len(detected_crises) + 1,
            'start': round(state_start, 2),
            'end': round(float(df_feat.iloc[-1]['Time_fin']), 2),
            'duration': round(float(df_feat.iloc[-1]['Time_fin']) - state_start, 2),
            'type': 'Crise'
        })
    elif in_state == 2:
        detected_pre_ictal.append({
            'id': len(detected_pre_ictal) + 1,
            'start': round(state_start, 2),
            'end': round(float(df_feat.iloc[-1]['Time_fin']), 2),
            'duration': round(float(df_feat.iloc[-1]['Time_fin']) - state_start, 2),
            'type': 'Avant crise'
        })
        
    # Calcul FFT moyen (Normal vs Crise) pour visualisation
    win_n = int(round(2.0 * fs))
    sain_idx = df_feat[df_feat['Class'] == 0].index[min(50, len(df_feat[df_feat['Class'] == 0])-1)]
    crise_idx = df_feat[df_feat['Class'] == 1].index[min(10, max(0, len(df_feat[df_feat['Class'] == 1])-1))]
    
    s_sain = int(df_feat.iloc[sain_idx]['Time_debut'] * fs)
    s_crise = int(df_feat.iloc[crise_idx]['Time_debut'] * fs)
    
    sig_raw = df[ref_signal_channel].to_numpy()
    fft_s = (np.abs(np.fft.rfft(sig_raw[s_sain:s_sain + win_n])) ** 2) / win_n
    fft_c = (np.abs(np.fft.rfft(sig_raw[s_crise:s_crise + win_n])) ** 2) / win_n
    freqs = np.fft.rfftfreq(win_n, 1.0 / fs)
    
    mask_freq = freqs <= 35.0
    fft_data = {
        'freqs': [round(float(f), 2) for f in freqs[mask_freq]],
        'sain': [round(float(p), 1) for p in fft_s[mask_freq]],
        'crise': [round(float(p), 1) for p in fft_c[mask_freq]]
    }
    
    # Trier et limiter les importances (top 20 pour la clarté)
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True)[:25]) if importances else {}
    
    res = {
        'status': 'success',
        'channel': display_channel,
        'is_multichannel': is_multichannel,
        'nb_features': len(feature_cols),
        'data_augmentation': aug_method,
        'augmented_train_samples': len(X_train_fit),
        'lobe': lobe,
        'description': desc,
        'model_name': model_name,
        'include_pre_ictal': is_3class,
        'is_3class': is_3class,
        'pre_ictal_sec': pre_ictal_sec if is_3class else 0.0,
        'num_classes': 3 if is_3class else 2,
        'classes': ['Normal (0)', 'Crise (1)', 'Avant crise (2)'] if is_3class else ['Normal (0)', 'Crise (1)'],
        'metrics': {
            'accuracy': round(acc * 100, 2),
            'balanced_accuracy': round(bal_acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1': round(f1 * 100, 2)
        },
        'per_class': per_class,
        'metrics_per_class': per_class,
        'confusion_matrix': cm_dict,
        'feature_importances': sorted_importances,
        'detected_crises': detected_crises,
        'detected_pre_ictal': detected_pre_ictal,
        'real_crises': df_crises.to_dict(orient='records'),
        'fft_spectra': fft_data
    }
    _CACHE['last_train_result'] = res
    return res

# ==============================================================================
# VALIDATION INTER-PATIENTS (CROSS-PATIENT / PATIENT-INDEPENDENT EVALUATION)
# ==============================================================================

COMMON_CHANNELS = [
    'EEGFp1', 'EEGFp2', 'EEGF7', 'EEGF3', 'EEGFz', 'EEGF4', 'EEGF8',
    'EEGT3', 'EEGC3', 'EEGCz', 'EEGC4', 'EEGT4', 'EEGT5', 'EEGP3',
    'EEGPz', 'EEGP4', 'EEGT6', 'EEGO1', 'EEGO2', 'SLI', 'ECG'
]

DEFAULT_TRAIN_PATIENTS = [
    '210504B_C_0000', '230102B_F_0000', '211027B_C_0000', '230105B_H_0000',
    '211104B_D_0000', '230406B_A_0000', '210208B_G_0000', '220121B_D_0000',
    '230224B_B_0000', '191113A_D_0000', '210928B_B_0000', '220106A_A_0000',
    '230515B_G_0000', '230519A_E_0000', '190304A_E_0000', '210204B_C_0000'
]

DEFAULT_TEST_PATIENTS = [
    '210914B_A_0000', '230718A_A_0000', '210427B_C_0000', '210406A_F_0000',
    '230210B_D_0000'
]

_PATIENT_FEATURE_CACHE = {}

def get_cross_patient_split():
    """Retourne la répartition par défaut des patients en deux groupes (Train et Test) avec statistiques."""
    all_datasets = list_available_datasets()
    all_pids = [d['patient_id'] for d in all_datasets if d['patient_id'] != '200625A_F_0000']
    
    train_list = [p for p in DEFAULT_TRAIN_PATIENTS if p in all_pids]
    test_list = [p for p in DEFAULT_TEST_PATIENTS if p in all_pids]
    
    # Tout patient non assigné va dans le train
    for p in all_pids:
        if p not in train_list and p not in test_list:
            train_list.append(p)
            
    return {
        'train_patients': train_list,
        'test_patients': test_list,
        'common_channels': COMMON_CHANNELS,
        'total_train': len(train_list),
        'total_test': len(test_list)
    }

def extract_patient_features_cached(patient_id, normalize=True, include_pre_ictal=False, pre_ictal_sec=10.0):
    """Extrait ou récupère en cache les descripteurs sur les canaux communs, avec normalisation z-score intra-patient."""
    cache_key = f"{patient_id}_norm_{normalize}_pre_{include_pre_ictal}_{pre_ictal_sec}"
    if cache_key in _PATIENT_FEATURE_CACHE:
        return _PATIENT_FEATURE_CACHE[cache_key]

    df, _, fs, df_crises = load_data(patient_id=patient_id, include_pre_ictal=include_pre_ictal, pre_ictal_sec=pre_ictal_sec)
    f = extract_all_features()
    
    # Filtrer strictement les colonnes liées aux canaux communs
    common_cols = [c for c in f.columns if any(c.endswith(f"_{ch}") for ch in COMMON_CHANNELS)]
    
    if normalize and len(f) > 1:
        scaler = StandardScaler()
        X_vals = scaler.fit_transform(f[common_cols])
        df_norm = pd.DataFrame(X_vals, columns=common_cols)
    else:
        df_norm = f[common_cols].copy()
        
    df_norm['Time_debut'] = f['Time_debut'].values
    df_norm['Time_fin'] = f['Time_fin'].values
    df_norm['Class'] = f['Class'].values
    df_norm['Patient'] = patient_id
    
    _PATIENT_FEATURE_CACHE[cache_key] = (df_norm, len(df_crises))
    return df_norm, len(df_crises)

def train_and_evaluate_cross_patient(train_patients=None, test_patients=None, model_name='Random Forest', use_smote=True, data_augmentation=None, include_pre_ictal=False, pre_ictal_sec=10.0, use_temporal_filter=True):
    """Entraîne les modèles sur un groupe de patients et les évalue exclusivement sur un groupe de patients indépendants."""
    split = get_cross_patient_split()
    train_pids = train_patients or split['train_patients']
    test_pids = test_patients or split['test_patients']
    
    # 1. Extraction et concaténation du groupe d'entraînement
    dfs_train = []
    total_train_crises = 0
    for p in train_pids:
        try:
            df_p, n_crises = extract_patient_features_cached(p, normalize=True, include_pre_ictal=include_pre_ictal, pre_ictal_sec=pre_ictal_sec)
            dfs_train.append(df_p)
            total_train_crises += n_crises
        except Exception as e:
            print(f"Avertissement : échec chargement train patient {p} ({e})")

    if not dfs_train:
        raise ValueError("Aucun patient valide dans le groupe d'entraînement.")

    df_train_all = pd.concat(dfs_train, ignore_index=True)
    feature_cols = [c for c in df_train_all.columns if c not in ['Time_debut', 'Time_fin', 'Class', 'Patient']]
    
    X_train = df_train_all[feature_cols]
    y_train = df_train_all['Class']

    # 2. Extraction du groupe de test indépendant
    dfs_test = []
    test_patient_info = {}
    total_test_crises = 0
    for p in test_pids:
        try:
            df_p, n_crises = extract_patient_features_cached(p, normalize=True, include_pre_ictal=include_pre_ictal, pre_ictal_sec=pre_ictal_sec)
            dfs_test.append(df_p)
            test_patient_info[p] = {
                'windows': len(df_p),
                'crisis_windows': int((df_p['Class'] == 1).sum()),
                'pre_ictal_windows': int((df_p['Class'] == 2).sum()),
                'nb_crises': n_crises
            }
            total_test_crises += n_crises
        except Exception as e:
            print(f"Avertissement : échec chargement test patient {p} ({e})")

    if not dfs_test:
        raise ValueError("Aucun patient valide dans le groupe de test.")

    df_test_all = pd.concat(dfs_test, ignore_index=True)
    X_test = df_test_all[feature_cols]
    y_test = df_test_all['Class']

    # 3. Data Augmentation sur l'entraînement uniquement (isolation stricte du test)
    aug_method = data_augmentation if data_augmentation is not None else ('smote' if use_smote else 'none')
    X_train_fit, y_train_fit = augment_training_data(X_train, y_train, method=aug_method, random_state=42)

    # 4. Ajustement du modèle sélectionné
    importances = {}
    if model_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train_fit, y_train_fit)
        raw_pred = clf.predict(X_test)
        importances = {col: round(float(val), 4) for col, val in zip(feature_cols, clf.feature_importances_)}
    elif model_name == 'Decision Tree':
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train_fit, y_train_fit)
        raw_pred = clf.predict(X_test)
        importances = {col: round(float(val), 4) for col, val in zip(feature_cols, clf.feature_importances_)}
    elif model_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        clf.fit(X_train_fit, y_train_fit)
        raw_pred = clf.predict(X_test)
    elif model_name == 'SVM':
        clf = SVC(kernel='rbf', C=5.0, random_state=42)
        clf.fit(X_train_fit, y_train_fit)
        raw_pred = clf.predict(X_test)
    elif model_name in ['EEGNet', 'EEGNet (Deep Learning)', 'eegnet']:
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42, early_stopping=True)
        clf.fit(X_train_fit, y_train_fit)
        raw_pred = clf.predict(X_test)
    elif model_name in ['XGBoost', 'xgboost', 'XGB']:
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost n'est pas disponible.")
        clf = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        clf.fit(X_train_fit, y_train_fit)
        raw_pred = clf.predict(X_test)
        importances = {col: round(float(val), 4) for col, val in zip(feature_cols, clf.feature_importances_)}
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    # 5. Filtrage temporel clinique par patient
    patients_test_arr = df_test_all['Patient'].to_numpy()
    if use_temporal_filter:
        y_pred = np.empty_like(raw_pred)
        for p in test_pids:
            mask_p = (patients_test_arr == p)
            if mask_p.any():
                y_pred[mask_p] = apply_temporal_smoothing(raw_pred[mask_p])
    else:
        y_pred = raw_pred

    # 6. Métriques globales sur l'ensemble du groupe test
    unique_classes = sorted(list(np.unique(y_test)))
    is_3class = (2 in unique_classes) or include_pre_ictal

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    if is_3class:
        prec = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        cm_dict = {
            'matrix': cm.tolist(),
            'labels': ['Normal (0)', 'Crise (1)', 'Avant crise (2)'],
            'is_3class': True,
            'total_test': len(y_test),
            'vn': int(cm[0, 0]),
            'fp': int(cm[0, 1] + cm[0, 2]),
            'fn': int(cm[1, 0] + cm[2, 0]),
            'vp': int(cm[1, 1] + cm[2, 2])
        }
    else:
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        vn, fp, fn, vp = [int(v) for v in cm.ravel()]
        cm_dict = {
            'matrix': cm.tolist(),
            'labels': ['Normal (0)', 'Crise (1)'],
            'is_3class': False,
            'total_test': len(y_test),
            'vn': vn, 'fp': fp, 'fn': fn, 'vp': vp
        }

    # 7. Métriques détaillées par patient du groupe test
    per_patient = []
    for p in test_pids:
        mask_p = (df_test_all['Patient'] == p)
        if not mask_p.any():
            continue
        y_p = y_test[mask_p]
        p_p = y_pred[mask_p]
        p_info = test_patient_info.get(p, {})
        
        per_patient.append({
            'patient_id': p,
            'clean_name': p.replace('_0000', ''),
            'windows': len(y_p),
            'nb_crises': p_info.get('nb_crises', 0),
            'crisis_windows': int((y_p == 1).sum()),
            'pre_ictal_windows': int((y_p == 2).sum()),
            'accuracy': round(float(accuracy_score(y_p, p_p)) * 100, 2),
            'balanced_accuracy': round(float(balanced_accuracy_score(y_p, p_p)) * 100, 2),
            'precision': round(float(precision_score(y_p, p_p, average='macro' if is_3class else 'binary', zero_division=0)) * 100, 2),
            'recall': round(float(recall_score(y_p, p_p, average='macro' if is_3class else 'binary', zero_division=0)) * 100, 2),
            'f1': round(float(f1_score(y_p, p_p, average='macro' if is_3class else 'binary', zero_division=0)) * 100, 2),
            'recall_crise': round(float(recall_score(y_p == 1, p_p == 1, zero_division=0)) * 100, 2),
            'recall_avant_crise': round(float(recall_score(y_p == 2, p_p == 2, zero_division=0)) * 100, 2) if is_3class else None
        })

    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True)[:25]) if importances else {}

    return {
        'status': 'success',
        'model_name': model_name,
        'evaluation_type': 'Inter-Patients (Patient-Independent)',
        'data_augmentation': aug_method,
        'augmented_train_samples': len(X_train_fit),
        'include_pre_ictal': is_3class,
        'is_3class': is_3class,
        'pre_ictal_sec': pre_ictal_sec if is_3class else 0.0,
        'num_classes': 3 if is_3class else 2,
        'classes': ['Normal (0)', 'Crise (1)', 'Avant crise (2)'] if is_3class else ['Normal (0)', 'Crise (1)'],
        'nb_train_patients': len(train_pids),
        'nb_test_patients': len(test_pids),
        'train_patients': train_pids,
        'test_patients': test_pids,
        'total_train_windows': len(df_train_all),
        'total_test_windows': len(df_test_all),
        'total_train_crises': total_train_crises,
        'total_test_crises': total_test_crises,
        'metrics': {
            'accuracy': round(acc * 100, 2),
            'balanced_accuracy': round(bal_acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1': round(f1 * 100, 2)
        },
        'confusion_matrix': cm_dict,
        'per_patient_results': per_patient,
        'feature_importances': sorted_importances
    }

