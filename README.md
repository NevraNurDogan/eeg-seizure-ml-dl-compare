
#  EEG Sinyalleri Kullanarak Epileptik Nöbet Tespiti: Klasik ML ve Uçtan Uca Derin Öğrenme Karşılaştırması

Bu depo, Elektroensefalografi (EEG) sinyallerinden epileptik nöbet anlarını tespit etmek amacıyla Geleneksel Makine Öğrenmesi (Özellik Çıkarımı) ve Uçtan Uca (End-to-End) Derin Öğrenme yaklaşımlarını karşılaştıran bir akademik projenin kaynak kodlarını ve detaylı analizlerini içermektedir.



##  Proje Özeti

Epilepsi, dünya çapında milyonlarca insanı etkileyen yaygın bir nörolojik bozukluktur. Erken ve doğru nöbet tespiti, hastaların yaşam kalitesi ve tedavi yönetimleri için hayati önem taşır. Bu çalışma, UCI "Epileptic Seizure Recognition" veri setini kullanarak, bir EEG segmentinin nöbet içerip içermediğini ikili (binary) olarak sınıflandıran dört farklı modeli incelemektedir:

* **Random Forest (RF)** - Özellik Çıkarımlı (12 Özellik)
* **Support Vector Machine (SVM)** - Özellik Çıkarımlı (12 Özellik)
* **1D-CNN (Özellik Çıkarımlı)** - Sadece 12 Özellik üzerinde eğitilmiş CNN.
* **1D-CNN (Ham Veri)** - 178 veri noktalık ham zaman serisi üzerinde eğitilmiş **Uçtan Uca CNN**.

Proje, özellikle **Özellik Mühendisliği (Feature Engineering)** adımının gerekliliğini ve **Uçtan Uca Derin Öğrenme** yaklaşımının ham EEG verisi üzerindeki etkinliğini karşılaştırmayı hedeflemiştir.

---

##  Metodoloji ve Modeller

Çalışma, iki ana metodolojik yaklaşımı karşılaştırmıştır:

### 1. Özellik Çıkarımlı Yaklaşımlar (RF, SVM, 1D-CNN-Özellik)

Her 1 saniyelik EEG segmentinden 12 adet istatistiksel ve spektral özellik (Ortalama, Standart Sapma, Frekans Bant Güçleri: Delta, Teta, Alfa, Beta, Gama vb.) çıkarılmıştır.

* **Klasik ML:** RF ve RBF çekirdekli SVM, bu 12 özellik uzayı üzerinde eğitilmiştir.
* **Derin Öğrenme:** Basitleştirilmiş bir 1D-CNN, aynı 12 özellik üzerinde eğitilerek klasik ML ile adil bir karşılaştırma sağlanmıştır.

### 2. Uçtan Uca Ham Veri Yaklaşımı (1D-CNN-Ham Veri)

Bu yaklaşım, özellik çıkarma adımını atlayarak **178 veri noktalık ham sinyali** doğrudan girdi olarak kabul eden ve hiyerarşik olarak kendi özelliklerini öğrenen derin ve optimize edilmiş bir 1D-CNN mimarisi kullanmıştır.



---

## Ana Sonuçlar ve Değerlendirme

Test seti (2300 örnek) üzerinde yapılan değerlendirmede, modeller arasında önemli bir performans ödünleşmesi (trade-off) gözlemlenmiştir.

| Model Tipi | Yaklaşım | F1-Skoru | Sensitivity (Recall) | Specificity |
| :--- | :--- | :--- | :--- | :--- |
| **1D-CNN (Ham Veri)** | Uçtan Uca | **0.9667** | 0.9457 | **0.9973** |
| Random Forest (RF) | Özellik Çıkarımlı | 0.9598 | 0.9609 | 0.9842 |
| 1D-CNN (Özellik) | Özellik Çıkarımlı | 0.9520 | **0.9696** | 0.9693 |

### Tartışma

* **En Yüksek Genel Başarı (F1-Skoru):** **Uçtan Uca 1D-CNN (Ham Veri)** modeli, genel performans metriklerinde en yüksek F1-Skorunu (0.9667) elde etmiştir.
* **En Az Yanlış Alarm (Specificity):** Ham Veri modeli, 1840 normal vakanın yalnızca 5'ini yanlış alarm olarak sınıflandırarak (Specificity: 0.9973) **klinik kullanım için en güvenilir** yanlış alarm oranını sunmuştur.
* **En Az Kaçırılan Nöbet (Sensitivity):** **1D-CNN (Özellik Çıkarımlı)** modeli, nöbet vakalarını yakalamada (Sensitivity: 0.9696) en başarılı olmuştur (sadece 14 False Negative). Bu, nöbetlerin kaçırılmasının kritik olduğu senaryolarda tercih edilebilir.

---

##  Kurulum ve Kullanım

### Gereksinimler

Bu projeyi yerel ortamınızda çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

```bash
pip install pandas scikit-learn tensorflow keras scipy matplotlib seaborn
```
Veri Seti
Veri Seti: Projede kullanılan veri seti, Kaggle platformunda yayınlanan "Epileptic Seizure Recognition" veri setidir.

Lütfen bu veri setini indirip projenin ana dizinine yerleştirin.

Çalıştırma
Projenin temel adımları (Veri Ön İşleme, Özellik Çıkarımı, Model Eğitimi, Değerlendirme) aşağıdaki ana Python dosyalarında bulunmaktadır:

data_preprocessing.py: Veri seti yükleme ve binary sınıflandırmaya dönüştürme adımları.

feature_engineering.py: 12 adet zaman ve frekans alanı özelliğinin çıkarılması.

ml_models.py: RF ve SVM modellerinin eğitilmesi ve değerlendirilmesi.

cnn_models.py: 1D-CNN (Özellik) ve 1D-CNN (Ham Veri) mimarilerinin oluşturulması, eğitilmesi ve değerlendirilmesi.

🔗 Kaynakça
Veri Seti (Kullanılan): Epileptic Seizure Recognition Dataset

Veri Seti (Orijinal Kaynak): Andrzejak RG, Lehnertz K, Rieke C, Mormann F, David P, Elger CE (2001) Indications of nonlinear deterministic and finite dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state, Phys. Rev. E, 64, 061907.
