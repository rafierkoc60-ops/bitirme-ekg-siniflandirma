# MIT-BIH Aritmi Veri Tabanında Sınıf Dengesizliği ve Hasta Bazlı Değerlendirme

**Öğrenci:** Rafi Erkoç (223204038)  
**Akademik Danışman:** Arş. Gör. Dr. Meral Kılıçarslan Ouach  
**Ders:** Bitirme Projesi 1 (2026-2027 Güz)  
**Bölüm:** Elektrik-Elektronik Mühendisliği  
**Hedef Çağrı:** TÜBİTAK 2242 / TEKNOFEST 2027 (0 TL bütçe, açık veri tabanlı yazılım projesi)

---

## 📌 Proje Özeti
Bu çalışmada, MIT-BIH Aritmi veri tabanındaki sınıf dengesizliğinin (özellikle N sınıfının baskınlığı, S ve V sınıflarının azlığı) Random Forest sınıflandırıcısı üzerindeki etkisi, hasta sızıntısını önleyen Chazal hasta-bazlı ayrımı (DS1 Eğitim, DS2 Test) ile incelenmiştir. Dört temel öznitelik (önceki RR, sonraki RR, yerel oran ve R-tepe genliği) kullanılmıştır.

---

## 📊 Deneysel Sonuçlar (DS2 - 49.683 Atım Test Kümesi)

| Model | Makro-F1 | N Duyarlılık (Recall) | S Duyarlılık (Recall) | V Duyarlılık (Recall) | F Duyarlılık (Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Temel Model (Dengesiz)** | **0.460** | 0.957 | 0.022 | 0.753 | 0.106 |
| **2. Sınıf Ağırlıklı (Class-Weight)** | 0.447 | 0.949 | 0.019 | 0.743 | 0.077 |
| **3. Aşırı Örnekleme (Random Oversampling - Yalnızca DS1)** | 0.444 | 0.941 | 0.025 | 0.776 | 0.038 |

---

## 🔍 Tartışma ve Dürüst Değerlendirme
- **Dengeleme Neden Makro-F1'i Artırmadı?**  
  Eğitim kümesine uygulanan sınıf ağırlıklandırma ve aşırı örnekleme yöntemleri makro-F1 skorunu artırmamış (0.460'tan 0.447 ve 0.444'e gerilemiş), Supraventriküler (S) sınıfı duyarlılığı %2 civarında kalmıştır. Bunun nedeni, yalnızca 4 zamansal/genliksel özniteliğin (RR aralıkları ve R genliği) atriyal kökenli S atımları ile normal N atımlarını ayırt etmede yetersiz kalmasıdır. Morfolojik P dalgası bilgisi olmaksızın sınıf ağırlığı artırıldığında, model N atımlarını yanlışlıkla S olarak etiketlemekte ve sahte pozitifler nedeniyle kesinlik (precision) düşmektedir. Ventriküler (V) sınıfı ise belirgin morfolojik genişliği ve kompanse edici duraklaması sayesinde %75-%77 duyarlılıkla başarıyla ayrışmaktadır.

---

## 📁 Depo İçeriği
- `ekg_siniflandirma_modelleri.py`: DS1/DS2 ayrımı, model eğitimleri ve CSV üretim scripti.
- `karsilastirma_tablosu.csv`: 3 modelin tüm sınıf metriklerini içeren sonuç dosyası.
- `karsilastirma_matrisleri.png`: 3 modele ait normalize karmaşıklık matrisleri.
- `Bitirme_Projesi_Rapor_Taslagi.pdf`: MÜDEK standartlarına uygun dönem proje raporu taslağı.
