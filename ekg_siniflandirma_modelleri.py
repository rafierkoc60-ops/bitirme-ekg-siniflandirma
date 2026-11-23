"""
=============================================================================
BİTİRME PROJESİ 1-2 (2026-2027 GÜZ)
Proje Başlığı: EKG Atım Sınıflandırmasında Nadir Aritmi Sınıflarının Başarısını
               Sınıf Dengeleme Yöntemleriyle İyileştirme
Öğrenci: Rafi Erkoç (223204038)
Danışman: Arş. Gör. Dr. Meral Kılıçarslan Ouach
Kullanılan Araçlar: Python 3, Scikit-Learn, Imbalanced-Learn, WFDB, Pandas, NumPy
=============================================================================
"""

import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import RandomOverSampler

# 1. AAMI EC57 Standart 4 Ana Sınıf Eşlemesi
AAMI_MAPPING = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',          # Normal grubu
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',                     # Supraventriküler ektopik (Nadir)
    'V': 'V', 'E': 'V',                                         # Ventriküler ektopik (Nadir)
    'F': 'F'                                                    # Füzyon (Çok nadir)
}

# 2. de Chazal et al. (2004) Hasta-Bazlı (Inter-Patient) Ayrım
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
    print("=== 1. VERİ YÜKLEME VE HASTA-BAZLI ÖZNİTELİK ÇIKARIMI ===")
    X_train, y_train = extract_features(DS1_TRAIN)
    X_test, y_test = extract_features(DS2_TEST)
    classes = ['N', 'S', 'V', 'F']

    print(f"DS1 Eğitim Atım Sayısı: {len(X_train)}")
    print(f"DS2 Test Atım Sayısı  : {len(X_test)} (Test kümesine veri sızıntısı YOK!)")

    # Model 1: Temel Model (Dengeleme Yok)
    print("\n[1/3] Temel Model (Baseline) eğitiliyor...")
    m1 = RandomForestClassifier(n_estimators=100, random_state=42)
    m1.fit(X_train, y_train)
    y_pred_m1 = m1.predict(X_test)

    # Model 2: Sınıf Ağırlıklı Model
    print("[2/3] Sınıf Ağırlıklı Model (Class-Weight) eğitiliyor...")
    m2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    m2.fit(X_train, y_train)
    y_pred_m2 = m2.predict(X_test)

    # Model 3: Yalnız Eğitimde Aşırı Örnekleme
    print("[3/3] Yalnız Eğitim Kümesinde Aşırı Örnekleme (RandomOverSampler) uygulanıyor...")
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    m3 = RandomForestClassifier(n_estimators=100, random_state=42)
    m3.fit(X_train_res, y_train_res)
    y_pred_m3 = m3.predict(X_test)

    # Karşılaştırma Metrikleri ve CSV Üretimi (Hocanın İstediği Satırlar)
    def get_metrics(y_true, y_pred, name):
        rep = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
        res = {'Model': name}
        for c in classes:
            res[f'{c}_Recall'] = round(rep[c]['recall'], 3)
            res[f'{c}_Precision'] = round(rep[c]['precision'], 3)
            res[f'{c}_F1'] = round(rep[c]['f1-score'], 3)
        res['Makro_F1'] = round(f1_score(y_true, y_pred, average='macro', zero_division=0), 3)
        return res

    results = [
        get_metrics(y_test, y_pred_m1, '1. Temel Model (Baseline)'),
        get_metrics(y_test, y_pred_m2, '2. Sınıf Ağırlıklı (Class-Weight)'),
        get_metrics(y_test, y_pred_m3, '3. Aşırı Örnekleme (Oversampled)')
    ]

    df_comp = pd.DataFrame(results)
    csv_file = 'karsilastirma_tablosu.csv'
    df_comp.to_csv(csv_file, index=False, sep=';', encoding='utf-8-sig')
    print(f"\nKarşılaştırma CSV dosyası başarıyla üretildi: {csv_file}")
    print(df_comp[['Model', 'S_Recall', 'S_F1', 'V_Recall', 'V_F1', 'Makro_F1']])

if __name__ == '__main__':
    main()
