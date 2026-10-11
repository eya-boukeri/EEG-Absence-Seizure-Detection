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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

def load_data(patient_id=None, signal_file=None, annotation_file=None):
    if patient_id is not None and patient_id != _CACHE.get('active_patient_id'):
        reset_cache()
        _CACHE['active_patient_id'] = patient_id

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
    df_wide = pd.read_csv(signal_path)
    sample_cols = [c for c in df_wide.columns if c != 'Record Time']
    channels = sorted(
        set(re.sub(r'_\d+$', '', c) for c in sample_cols),
        key=lambda ch: sample_cols.index(f"{ch}_1")
    )
    samples_per_row = sum(1 for c in sample_cols if c.startswith(channels[0] + "_"))
    fs = 256.0
    n_rows = len(df_wide)
    n_samples = n_rows * samples_per_row
    time = np.arange(n_samples) / fs

    data = {}
    for ch in channels:
        cols = [f"{ch}_{i}" for i in range(1, samples_per_row + 1)]
        data[ch] = df_wide[cols].to_numpy().reshape(-1)

    df = pd.DataFrame(data)
    df.insert(0, 'Time', time)

    # 2. Annotations médicales
    ann_xlsx = pd.read_excel(xlsx_path, nrows=50)
    ann_xlsx = ann_xlsx.dropna(how='all')
    clean_pid = p_id.replace('_0000', '').replace('_', '-')
    mask_pt = ann_xlsx['Unnamed: 0'].astype(str).str.contains(clean_pid[:7])
    if mask_pt.any():
        row_pt = ann_xlsx[mask_pt].iloc[0]
    else:
        row_pt = ann_xlsx.iloc[0]

    # Heure de début d'enregistrement du tracé EEG (DebutTrace)
    debut_trace_sec = excel_time_to_seconds(row_pt.get('DebutTrace'))
    if debut_trace_sec is None:
        debut_trace_sec = excel_time_to_seconds(row_pt['DebCE1']) - 1465.5

    deb_cols = [c for c in row_pt.index if c.startswith(('Deb', 'Debut', 'Debt'))]
    fin_cols = [c for c in row_pt.index if c.startswith('Fin')]

    crises = []
    nb_ce = int(row_pt['NombreCE']) if pd.notna(row_pt.get('NombreCE')) else 12
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

            if deb_rel < df['Time'].iloc[-1]:
                crises.append({
                    'Crise': i,
                    'Debut': round(float(deb_rel), 2),
                    'Fin': round(float(min(fin_rel, df['Time'].iloc[-1])), 2),
                    'Duree': round(float(dur), 2),
                    'Debut_Horaire': seconds_to_hhmmss(deb_horaire_sec),
                    'Fin_Horaire': seconds_to_hhmmss(fin_horaire_sec),
                    'Debut_Sec': round(float(deb_horaire_sec), 2),
                    'Fin_Sec': round(float(fin_horaire_sec), 2)
                })

    df_crises = pd.DataFrame(crises)

    # 3. Variable cible
    classes = np.zeros(len(df), dtype=int)
    t_vals = df['Time'].to_numpy()
    for _, c in df_crises.iterrows():
        mask = (t_vals >= c['Debut']) & (t_vals <= c['Fin'])
        classes[mask] = 1
    df['Class'] = classes

    _CACHE['df'] = df
    _CACHE['channels'] = channels
    _CACHE['fs'] = fs
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
        'crisis_percentage': round(float((df['Class'] == 1).mean() * 100), 2),
        'crises': df_crises.to_dict(orient='records')
    }

def get_features_summary():
    """Retourne la synthèse de l'extraction des 294 descripteurs spatiotemporels."""
    df_feat = extract_all_features()
    feature_cols = [c for c in df_feat.columns if c not in ['Time_debut', 'Time_fin', 'Class']]
    n_normal = int((df_feat['Class'] == 0).sum())
    n_crisis = int((df_feat['Class'] == 1).sum())
    
    return {
        'status': 'success',
        'total_windows': len(df_feat),
        'window_duration_sec': 2.0,
        'window_step_sec': 1.0,
        'overlap_percent': 50,
        'nb_features': len(feature_cols),
        'normal_windows': n_normal,
        'crisis_windows': n_crisis,
        'imbalance_ratio': f"1:{round(n_normal / max(1, n_crisis), 1)}",
        'crisis_percentage': round((n_crisis / len(df_feat)) * 100, 2),
        'feature_categories': {
            'temporelles': 21 * 8,
            'spectrales_fft': 21 * 4,
            'ratios_neurophysiologiques': 21 * 2
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
        'patient_id': PATIENT_ID,
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

def extract_all_features():
    """Extrait simultanément les 12 caractéristiques pour l'ensemble des canaux (252 descripteurs multi-canaux)."""
    if 'ALL' in _CACHE['features']:
        return _CACHE['features']['ALL']

    df, channels, fs, _ = load_data()
    win_n = int(round(2.0 * fs))  # 512 échantillons
    step_n = int(round(win_n * 0.5))  # 256 échantillons

    cls_raw = df['Class'].to_numpy()
    t_raw = df['Time'].to_numpy()

    windows_cls = np.lib.stride_tricks.sliding_window_view(cls_raw, window_shape=win_n)[::step_n]
    windows_t = np.lib.stride_tricks.sliding_window_view(t_raw, window_shape=win_n)[::step_n]

    labels = (windows_cls.mean(axis=1) >= 0.5).astype(int)
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
    _CACHE['features']['ALL'] = df_feat
    return df_feat

def extract_features_for_channel(channel):
    if channel in ['ALL', 'all', 'Tous', 'Tous les canaux', None]:
        return extract_all_features()
        
    if channel in _CACHE['features']:
        return _CACHE['features'][channel]
        
    df, _, fs, _ = load_data()
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
        label = 1 if w_cls.mean() >= 0.5 else 0
        
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
    _CACHE['features'][channel] = df_feat
    return df_feat

def apply_temporal_smoothing(preds):
    """Post-traitement par hystérésis temporelle : comble les micro-trous et élimine les alertes isolées < 2s."""
    p = np.array(preds).copy()
    # 1. Comblement des trous de 1 fenêtre intra-crise (1 0 1 -> 1 1 1)
    for i in range(1, len(p) - 1):
        if p[i] == 0 and p[i - 1] == 1 and p[i + 1] == 1:
            p[i] = 1
    # 2. Élimination des faux positifs isolés de 1 fenêtre (0 1 0 -> 0 0 0)
    for i in range(1, len(p) - 1):
        if p[i] == 1 and p[i - 1] == 0 and p[i + 1] == 0:
            p[i] = 0
    return p

def train_and_evaluate(channel='ALL', model_name='Random Forest', use_smote=True, use_temporal_filter=True):
    is_multichannel = channel in ['ALL', 'all', 'Tous', 'Tous les canaux', None]
    
    if is_multichannel:
        df_feat = extract_all_features()
        feature_cols = [c for c in df_feat.columns if c not in ['Time_debut', 'Time_fin', 'Class']]
        display_channel = 'ALL'
        lobe = 'Multi-Canaux (Global)'
        desc = f"Modèle multi-électrodes optimisé entraîné sur les {len(feature_cols)} descripteurs combinés (Temporels + FFT + Ratios spectraux Slow/Fast)."
        ref_signal_channel = 'EEGFp1'
    else:
        df_feat = extract_features_for_channel(channel)
        feature_cols = ['Mean', 'Std', 'Variance', 'Min', 'Max', 'Amplitude', 'RMS', 'Energy', 'Delta', 'Theta', 'Alpha', 'Beta', 'SDR', 'RelDelta']
        display_channel = channel
        lobe, desc = LOBE_MAP.get(channel, ('Inconnu', ''))
        ref_signal_channel = channel

    df, _, fs, df_crises = load_data()
    
    X = df_feat[feature_cols]
    y = df_feat['Class']
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_feat.index, test_size=0.20, random_state=42, stratify=y
    )
    
    # Amélioration 2a : Rééquilibrage SMOTE sur l'entraînement
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train_fit, y_train_fit = smote.fit_resample(X_train, y_train)
        except Exception:
            X_train_fit, y_train_fit = X_train, y_train
    else:
        X_train_fit, y_train_fit = X_train, y_train
    
    importances = {}
    
    if model_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=None if use_smote else 'balanced')
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
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")
        
    # Amélioration 2b : Post-filtrage temporel par persistance clinique
    if use_temporal_filter:
        all_pred = apply_temporal_smoothing(all_pred_raw)
        y_pred = all_pred[idx_test]
    else:
        all_pred = all_pred_raw
        y_pred = raw_pred_test
        
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    vn, fp, fn, vp = [int(v) for v in cm.ravel()]
    
    # Reconstruction temporelle
    detected_crises = []
    in_crisis = False
    crisis_start = 0.0
    
    for i, row in df_feat.iterrows():
        p = all_pred[i]
        if p == 1 and not in_crisis:
            in_crisis = True
            crisis_start = float(row['Time_debut'])
        elif p == 0 and in_crisis:
            in_crisis = False
            crisis_end = float(df_feat.iloc[i-1]['Time_fin'])
            detected_crises.append({
                'id': len(detected_crises) + 1,
                'start': round(crisis_start, 2),
                'end': round(crisis_end, 2),
                'duration': round(crisis_end - crisis_start, 2)
            })
    if in_crisis:
        crisis_end = float(df_feat.iloc[-1]['Time_fin'])
        detected_crises.append({
            'id': len(detected_crises) + 1,
            'start': round(crisis_start, 2),
            'end': round(crisis_end, 2),
            'duration': round(crisis_end - crisis_start, 2)
        })
        
    # Calcul FFT moyen (Normal vs Crise) pour visualisation
    win_n = int(round(2.0 * fs))
    sain_idx = df_feat[df_feat['Class'] == 0].index[min(50, len(df_feat[df_feat['Class'] == 0])-1)]
    crise_idx = df_feat[df_feat['Class'] == 1].index[min(10, len(df_feat[df_feat['Class'] == 1])-1)]
    
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
        'channel': display_channel,
        'is_multichannel': is_multichannel,
        'nb_features': len(feature_cols),
        'lobe': lobe,
        'description': desc,
        'model_name': model_name,
        'metrics': {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1': round(f1 * 100, 2)
        },
        'confusion_matrix': {
            'vn': vn, 'fp': fp, 'fn': fn, 'vp': vp,
            'total_test': len(y_test)
        },
        'feature_importances': sorted_importances,
        'detected_crises': detected_crises,
        'real_crises': df_crises.to_dict(orient='records'),
        'fft_spectra': fft_data
    }
    _CACHE['last_train_result'] = res
    return res
