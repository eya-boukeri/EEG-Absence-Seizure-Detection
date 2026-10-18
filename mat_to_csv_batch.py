import os
import sys
import glob
import numpy as np
import pandas as pd

# Reconfigure stdout/stderr pour UTF-8 sur Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Vérification de matio et basculement automatique vers Python 3.13 si nécessaire
try:
    from matio import load_from_mat
except ImportError:
    # Si nous sommes sous un Python où matio n'est pas installé (ex: Python 3.14 sans wheels C),
    # et que Python 3.13 est disponible sur le système :
    py313_path = r"C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe"
    if os.path.exists(py313_path) and sys.executable.lower() != py313_path.lower():
        import subprocess
        print(f"Basculement automatique vers Python 3.13 ({py313_path}) pour le support mat-io...")
        cmd = [py313_path, __file__] + sys.argv[1:]
        res = subprocess.run(cmd)
        sys.exit(res.returncode)
    else:
        sys.exit("Erreur : Le package 'mat-io' n'est pas installé dans cet environnement.\n"
                 "Veuillez l'installer avec : pip install mat-io")


def convert(mat_path, csv_path):
    data = load_from_mat(mat_path, raw_data=False, add_table_attrs=True)
    table = list(data.values())[0]

    first_col = table.iloc[:, 0]
    is_signal = isinstance(first_col.iloc[0], np.ndarray)

    if is_signal:
        # --- fichier "_d" : signal EEG (timetable imbriquée) ---
        channels = table.columns.tolist()
        fs = len(np.asarray(table[channels[0]].iloc[0]).flatten())
        n_samples = len(table) * fs

        flat = {ch: np.concatenate([np.asarray(v).flatten() for v in table[ch].values])
                for ch in channels}
        df = pd.DataFrame(flat)
        df.insert(0, "Time", np.arange(n_samples) / fs)
        kind = "signal EEG"
    else:
        # --- fichier "_h" : journal d'événements ---
        df = table.reset_index()
        # Normalisation du nom de colonne Onset
        if "Onset" in df.columns:
            onset_col = "Onset"
        elif "Time" in df.columns:
            onset_col = "Time"
            df.rename(columns={"Time": "Onset"}, inplace=True)
        elif "index" in df.columns:
            onset_col = "index"
            df.rename(columns={"index": "Onset"}, inplace=True)
        else:
            onset_col = df.columns[0]
            df.rename(columns={onset_col: "Onset"}, inplace=True)

        if hasattr(df["Onset"].dt, "total_seconds"):
            df["Onset"] = df["Onset"].dt.total_seconds()

        # Si Duration est au format Timedelta, le convertir aussi en secondes
        if "Duration" in df.columns and hasattr(df["Duration"].dt, "total_seconds"):
            df["Duration"] = df["Duration"].dt.total_seconds()

        kind = "journal d'annotations"

    df.to_csv(csv_path, index=False)
    print(f"[{kind}] {os.path.basename(csv_path)} -> {df.shape[0]} lignes, {df.shape[1]} colonnes ({os.path.getsize(csv_path)/(1024*1024):.1f} Mo)")


def convert_folder(folder_path, outdir=None, skip_existing=False):
    if not os.path.isdir(folder_path):
        sys.exit(f"Erreur : dossier introuvable : {folder_path}")

    mat_files = sorted(glob.glob(os.path.join(folder_path, "*.mat")))
    if not mat_files:
        sys.exit(f"Aucun fichier .mat trouvé dans {folder_path}")

    outdir = outdir or folder_path
    os.makedirs(outdir, exist_ok=True)
    print(f"Trouvé {len(mat_files)} fichier(s) .mat dans {folder_path}")
    print(f"Dossier de sortie : {outdir}\n")

    failed = []
    skipped = []
    converted = []

    for mat_path in mat_files:
        base = os.path.splitext(os.path.basename(mat_path))[0]
        csv_path = os.path.join(outdir, f"{base}.csv")

        if skip_existing and os.path.exists(csv_path):
            print(f"  [Ignoré] {base}.csv existe déjà.")
            skipped.append(mat_path)
            continue

        try:
            convert(mat_path, csv_path)
            converted.append(mat_path)
        except Exception as e:
            print(f"  [Erreur] {mat_path}: échec conversion ({e})")
            failed.append((mat_path, str(e)))

    print(f"\n========================================================")
    print(f"Conversion terminée : {len(converted)} convertis, {len(skipped)} ignorés, {len(failed)} échecs.")
    print(f"========================================================")
    if failed:
        print("Fichiers en échec :")
        for f, err in failed:
            print(f"  - {f}: {err}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        outdir = folder_path
    else:
        folder_path = sys.argv[1]
        outdir = sys.argv[2] if len(sys.argv) > 2 else folder_path

    skip_existing = "--skip-existing" in sys.argv
    convert_folder(folder_path, outdir, skip_existing=skip_existing)
