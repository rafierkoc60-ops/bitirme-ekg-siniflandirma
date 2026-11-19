"""
Bitirme Projesi 1-2 (2026-2027 Guz)
EKG Sinyallerinde Sinif Dengesizligi Altinda Nadir Aritmi Siniflandirmasi
Ogrenci: Rafi Erkoc
Danisman: Ars. Gor. Ceylan Cicek

Kullanilan Araclar ve Kutuphaneler:
- Python 3
- Scikit-Learn (RandomForestClassifier)
- Imbalanced-Learn (RandomOverSampler)
- WFDB (Waveform Database library for PhysioNet)
- NumPy, Pandas, Matplotlib, Seaborn
"""

import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import RandomOverSampler

# 1. AAMI 4 ANA SINIF ESLEMESI
AAMI_MAPPING = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F'
}

# 2. HASTA-BAZLI (INTER-PATIENT) AYRIM (Chazal et al. Standardi)
DS1_TRAIN = ['101', '106', '108', '109', '112', '115', '118', '119', 
             '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']

DS2_TEST  = ['100', '103', '105', '111', '113', '117', '121', '123', 
             '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']

def extract_features(records):
    X, y = [], []
    for rec in records:
        try:
            header = wfdb.rdheader(rec, pn_dir='mitdb')
            channel = ['MLII'] if 'MLII' in header.sig_name else [header.sig_name[0]]
            record = wfdb.rdrecord(rec, pn_dir='mitdb', channel_names=channel)
            signal = record.p_signal.flatten()
            ann = wfdb.rdann(rec, 'atr', pn_dir='mitdb')
            
            samples, symbols = ann.sample, ann.symbol
            for i in range(1, len(samples) - 1):
                sym = symbols[i]
                if sym in AAMI_MAPPING:
                    r_idx = samples[i]
                    pre_rr = (r_idx - samples[i-1]) / 360.0
                    post_rr = (samples[i+1] - r_idx) / 360.0
                    local_rr_ratio = pre_rr / (post_rr + 1e-6)
                    amp = signal[r_idx]
                    X.append([pre_rr, post_rr, local_rr_ratio, amp])
                    y.append(AAMI_MAPPING[sym])
        except Exception as e:
            print(f"Hata {rec}: {e}")
    return np.array(X), np.array(y)

def main():
    print("--- 1. Oznitelikler Cikariliyor (DS1 Egitim & DS2 Test) ---")
    X_train, y_train = extract_features(DS1_TRAIN)
    X_test, y_test = extract_features(DS2_TEST)
    classes = ['N', 'S', 'V', 'F']

    print(f"Egitim Atim Sayisi: {len(X_train)} | Test Atim Sayisi: {len(X_test)}")
    print("NOT: Veri sizintisini onlemek icin dengeleme YALNIZCA egitim kumesine uygulanir.")

    # Model 1: Temel Model (Dengeleme Yok)
    print("\n[1/3] Temel Model (Baseline) egitiliyor...")
    m1 = RandomForestClassifier(n_estimators=100, random_state=42)
    m1.fit(X_train, y_train)
    y_pred_m1 = m1.predict(X_test)

    # Model 2: Sinif Agirlikli Model
    print("[2/3] Sinif Agirlikli Model (Class-Weight) egitiliyor...")
    m2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    m2.fit(X_train, y_train)
    y_pred_m2 = m2.predict(X_test)

    # Model 3: Yalniz Egitimde Asiri Ornekleme
    print("[3/3] Yalniz Egitim Kumesinde Asiri Ornekleme (Oversampling) uygulaniyor...")
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    m3 = RandomForestClassifier(n_estimators=100, random_state=42)
    m3.fit(X_train_res, y_train_res)
    y_pred_m3 = m3.predict(X_test)

    # Sonuclari Raporlama
    models = [('Temel Model', y_pred_m1), ('Sinif Agirlikli', y_pred_m2), ('Asiri Ornekleme', y_pred_m3)]
    for name, pred in models:
        f1 = f1_score(y_test, pred, average='macro')
        print(f"\n>> {name} Makro-F1: {f1:.3f}")

if __name__ == '__main__':
    main()