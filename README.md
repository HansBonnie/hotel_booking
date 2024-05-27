# <p style="text-align:center;">**HOTEL BOOKING DEMAND**</p>
# **Business Understanding**

## Context : 
Industri perhotelan menjadi salah satu sektor yang bertumbuh cukup pesat di Portugal. Pada tahun 2019 pendapatan tahunan sektor ini meningkat [7,4% y-o-y](https://www.statista.com/statistics/1058491/sales-revenue-tourism-sector-portugal/). Diantara tren positif ini masih terdapat satu masalah yang dialami oleh Hotel di Portugal yaitu tamu/pelanggan yang membatalkan pesanan (cancel) terutama pada detik-detik terakhir. 

Hotel di Portugal ingin ada sebuah machine learning yang cukup akurat untuk memberi prediksi bahwa apakah tamu atau pelanggan ini akan membatalkan pesananannya atau tidak. Oleh karena itu mereka menugaskan data scientist untuk mendapatkan insight terkait pelanggan yang cancel dan juga membuat machine learning untuk menyelesaikan masalah tersebut.

Target :

0 : No Cancel

1 : Canceled


## Problem Statement :
*Last minute cancelation* atau *late cancelation* sangat mengganggu dalam proyeksi revenue dari hotel tesebut. Hal ini disebabkan karena jika seorang pelanggan *no show (include Last minute cancelation atau late cancelation)*, pihak hotel mungkin akan menghadapi resiko kamar kosong yang tidak dapat dijual lagi. Dalam kasus ini hotel juga sudah melakukan pengeluaran untuk operasional terkait tamu tersebut, seperti membersihkan dan menyiapkan kamar, yang akhirnya terbuang percuma. Cancelation ini juga salah satu alasan mengapa overbooking terjadi, tetapi dalam kasus kali ini, kita berbicara tentang *Last minute cancelation* atau *late cancelation*.

Pada akhirnya kita bisa simpulkan bahwa *Last minute cancelation* atau *late cancelation* di hotel berarti kehilangan pendapatan dan bahkan mengalami kerugian. 

## Goals :

Maka berdasarkan permasalahan tersebut, hotel ingin memiliki kemampuan untuk memprediksi kemungkinan seorang tamu melakukan cancel atau tidak, sehingga dapat melakukan upaya untuk mengantisi dan mempertahankan tamu yang terindikasi akan melakukan cancel.

Dan juga, perusahaan ingin mengetahui tamu dengan kategori seperti apa dan faktor-faktor apa saja yang cenderung membuat tamu tidak membatalkan pesanannya, sehingga hotel dapat membuat program-program yang lebih tepat sasaran dalam mengurangi jumlah pelanggan yang cancel.


## Analytic Approach :

Dalam kasus kali ini kita akan mencoba menemukan kategori pengunjung seperti apa yang cenderung membatalkan pesanannya (cancel).

Kemudian kita akan membangun model klasifikasi yang akan membantu perusahaan untuk dapat memprediksi probabilitas seorang pelanggan akan berhenti cancel atau tidak.


## Metric Evaluation :
Fokus dalam permodelan ini adalah tamu yang membatalkan pesanannya (cancel), maka target yang digunakan adalah `is_canceled` dengan detail sebagai berikut:

Target :
- 0 : Melanjutkan pesanan (No Cancel)
- 1 : Membatalkan pesanan (Cancel)

Type 1 error : False Positive (pelanggan yang aktualnya No Cancel tetapi diprediksi Cancel)
Konsekuensi: Kemungkinan terjadinya Overbooking karena hotel sudah menjual kamar tersebut ke tamu lain.

Type 2 error : False Negative (pelanggan yang aktualnya Cancel tetapi diprediksi No Cancel)
Konsekuensi: Hotel kehilangan pendapatan

Untuk memberikan dampak yang dapat dirasakan oleh hotel secara kuantitatif, maka kita akan coba hitung dampak biaya berdasarkan asumsi berikut :

Salah satu parameter mengukur pendapatan hotel dapat diketahui dari Occupancy hotel tersebut. Perhitungan Occupancy terhitung dari awal tamu/pelanggan melakukan pesanan (pada beberapa hotel tidak perlu melakukan pembayaran awal untuk memastikan kamar kita). Dalam perhotelan terdapat istiah CPOR (Cost Per Occupied Room), dimana terdapat komponen biaya yang harus dikeluarkan dari setiap kamar yang terisi, dan *late cancelation* juga masih terhitung sebagai kamar terisi karena kamar tersebut tetap memerlukan pembersihan, alokasi sarapan dan fasilitas-fasilitas lain yang ada pada hotel tersebut.
 
Pada sebuah hotel yang memiliki 100 kamar dengan harga US$100/night memiliki tingkat occupancy sebesar 80%. Memiliki COPR sebesar [15%](https://tgandh.com/articles/hospitality/rate-vs-cpor-against-guest-satisfaction/) dan tingkat cancelation sebesar [10%](https://www.appointmentreminders.com/how-is-a-no-show-rate-calculated/).

- Proyeksi awal pendapatan hotel per malam = 100 kamar x 80% x US$100 = US$8,000
- COPR = 15% x US$8,000 = US$1,200
- Pendapatan yang hilang karena Cancelation = 10% x (80% x 100 kamar) x US$100 = US$800
- COPR yang terbuang = 10% x US$1,200 = US$120

Dari gambaran tersebut kita bisa lihat bahwa potensi kerugian hotel :
- Pendapatan Kamar yang hilang = US$800
- Pengeluaran yang terbuang sia-sia= US$120

Jadi kerugian yang diakibatkan oleh cancelation bukan 1 : 1 dengan harga kamar, tetapi dari setiap cancelation mengakibatkan kerugian **1,15 kali lipat** dibanding harga kamar, atau jika dalam nominal maka menjadi :
- harga rata-rata kamar hotel di Portugal : [$99](https://www.budgetyourtrip.com/hotels/portugal-PT)
- total kerugian tiap 1 tamu melakukan cancel = 1.15 x $99 = **$113.85** 

Berdasarkan dampaknya maka kita ingin agar model dapat mengurangi jumlah False Negative (pelanggan yang aktualnya cancel tetapi diprediksi No Cancel). Maka metric utama yang akan digunakan adalah **Recall** atau juga diketahui sebagai True Positive Rate (TPR).


# **Data Understanding**

**Attribute Information**

| Attribute | Data Type | Description |
| --- | --- | --- | 
|country |object | Country of origin of the guest |
|market_segement |object | Market segment designation |
|previous_cancellations |Integer | Number of previous bookings that were canceled by the customer |
|booking_changes |Integer |Number of changes made to the booking |
|deposit_type |object | Type of deposit made (No Deposit, Refundable, Non Refund) |
|days_in_waiting_list |Integer | Number of days the booking was in the waiting list |
|customer_type |object | Type of customer (Transient, Contract, Transient-Party, Group) |
|reserved_room_type |object | Type of reserved room |
|required_car_parking_space |Integer | Number of car parking spaces required |
|total_of_special_request |Integer | Number of special requests made |
|is_canceled |Integer | Reservation cancellation status (0 = not canceled, 1 = canceled) |

# **Data Preprocessing**

## Feature Engineering

Disni kita akan melakukan feature selection yang nantinya akan digunakan dalam modeling.

Adapun feature yang akan kita drop atau modifikasi adalah feature yang memiliki High Cardinality.
Melakukan drop atau modifikasi pada feature yang memiliki High Cardinality berguna untuk :
- mempermudah interpretasi
- mencegah terjadinya overfitting 
- meringankan  komputasi

Disini terlihat bahwa ada 2 kolom yang memiliki High Cardinality yaitu `country` dan `days_in_waiting_list`. Pada kolom `country` akan kurangi nilai uniknya dengan hanya menampilkan 10 negara dengan jumlah kemunculan paling sering dan negara lain yang tidak termasuk kedalam itu akan diganti dengan others, sedangkan kita akan mempertahankan kolom `days_in_waiting_list`

## Imbalance Data

Persentasi antara 0 : 76% dan 1 : 23% masih tergolong kedalam mild imbalance, meskipun begitu kita akan tetap menerapkan teknik ***SMOTE-NC(Synthetic Minority Over-sampling Technique for Nominal and Continuous)***.

Disini kita memilih menggnukan SMOTE-NC karena dalam dataset ini terdapat feature numeric(continous) dan juga feature categorical

<p align="center"><img src =https://www.kdnuggets.com/wp-content/uploads/wijaya_7_smote_variations_oversampling_16.jpg>

## Handling Outliers

Menggunakan winsorizing untuk `previous_cancellations`, `booking_changes`, `days_in_waiting_list` dan `required_car_parking_spaces`

## Encoding

1. `country`: Binary encoding.
1. `market_segment`: One-hot encoding.
1. `deposit_type`: One-hot encoding.
1. `customer_type`: One-hot encoding.
1. `reserved_room_type`: binary encoding.

Tahap preprocessing ini dimulai dari Handling Outlier dengan menggunakan metode ***Winsorizing***, kemudian masuk kedalam tahap encoding. Pada tahap encoding ini kita merubah dari bentuk kategorikal menjadi numerical, pada tahap ini kita menjalankan 2 metode secara bersamaan yaitu ***OneHot encoding dan Binary encoding***, dengan detail sebagai berikut :
- OneHot Encoding : `market_segment`, `deposit_type`, `customer_type`
- Binary Encoding : `reserved_room_type`,`country`

Setelah melakukan encoding kemudian akan dilakukan scaling menggunakan ***Robust Scaler***. Karena data kita berada dalam kedaan mild imbalance (Target 0: 76%, 1: 24%) maka kita akan handling imbalance menggunakan teknik ***SMOTE-NC(Synthetic Minority Over-sampling Technique for Nominal and Continuous)*** karena didalam kasus ini terdapat feature yang bersifat numerical dan categorical

# Modeling

Pada tahap ini kita akan define seluruh model yang akan kita coba pada project kali ini, penjelasan singkat dari model yang akan kita gunakana adalah sebagai berikut :

|Nama Model|Definition|
|---|---|
|Logistic Regresion|Merupakan model statistik yang digunakan untuk memprediksi probabilitas dari kategori biner terklasifikasi, berdasarkan satu atau lebih fitur independen.|
|K-Nearest Neighbor(KNN)|Algoritma yang digunakan untuk klasifikasi dan regresi yang berfungsi dengan mencari sejumlah k data titik terdekat (neighbors) yang sudah diklasifikasikan untuk menentukan kategori dari titik baru.|
|Decision Tree|Model prediktif yang menggunakan struktur pohon untuk mengambil keputusan berdasarkan fitur input. Setiap node internal mewakili tes pada suatu atribut, setiap cabang mewakili hasil tes, dan setiap daun mewakili label kelas atau nilai yang diprediksi.|
|Random Forest|Metode ensemble yang menggunakan banyak decision trees untuk meningkatkan akurasi prediksi. Setiap tree dibangun dari subset acak dari data, dan hasil akhir diambil dari agregasi hasil semua tree (biasanya melalui voting untuk klasifikasi atau rata-rata untuk regresi).|
|Adaptive Boosting (AdaBoost)|Metode boosting yang menggabungkan beberapa weak classifiers untuk membentuk strong classifier. Algoritma ini memberikan bobot lebih pada instance yang salah klasifikasi sehingga classifier berikutnya fokus pada kasus-kasus yang sulit.|
|Gradient Boost|Teknik boosting yang menggabungkan model prediktif lemah, biasanya decision trees, dengan cara yang iteratif dan bertahap untuk mengoptimalkan fungsi loss melalui gradiennya.|
|Categorical Boosting (CatBoost)|Algoritma boosting yang dirancang khusus untuk menangani data kategorikal dengan baik dan menghindari overfitting, serta memanfaatkan pengurutan observasi untuk meningkatkan kecepatan dan akurasi.|
|Extreme Gradient Boosting (XGBoost)| Implementasi optimisasi dari gradient boosting yang memiliki keunggulan dalam kecepatan, efisiensi, dan kinerja, sering digunakan dalam kompetisi machine learning.|
|Light Gradient Boosting Machine (LightGBM)| Algoritma boosting yang efisien dan cepat yang menggunakan teknik histogram-based learning untuk mengurangi kompleksitas waktu pelatihan, serta mendukung paralelisasi dan pemotongan data besar.|

**Logistic Regression** menunjukkan hasil yang konsisten baik pada train set, validasi set, maupun test. Ini menunjukkan bahwa model ini cukup baik dalam mendeteksi cancelation (class 1).

**Gradient Boost dan AdaBoot** juga menunjukkan hasil yang relatif stabil, meskipun recall-nya sedikit lebih rendah dibanding Logistic Regression. Model ini masih cukup baik dalam mendeteksi cancelation, sehingga kedua model ini akan kita coba juga untuk tuning kembali.

Sedangkan pada model lain (KNN, LightGBM, Catboost, XGBoost, Random Forest, Decision Tree) menunjukkan tanda-tanda overfitting yang signifikan. Recall pada train set sangat tinggi, tetapi turun drastis pada validasi set dan test set. 

**Kesimpulan :**

Berdasarkan hasil pengujian yang dilakukan pada seluruh jenis model maka kita akan mengambil 3 model dengan nilai tertinggi yaitu **Logistic Regression, Gradient Boost dan AdaBoost** untuk kita lakukan hyperparameter tuning.  

## *FINAL MODEL - Gradient Boost*
Model terbaik yang akan kita pakai adalah Gradient Boost yang telah dituning.

Gradient Boosting adalah teknik ensemble yang digunakan dalam machine learning untuk membangun model prediksi yang kuat dengan menggabungkan prediksi dari beberapa decision tree. Konsep dasar dari Gradient Boost adalah dengan membuat beberapa decision tree yang tidak dapat melakukan prediksi secara mandiri, dan kemudian digabungkan menggunakan teknik ensamble. Selama proses penggabungan akan ditemukan kesalahan yang kemudian model akan berusaha memperbaiki kesalahan dari model sebelumnya. Setiap model baru akan melakukan prediksi terhadap residu dari model sebelumnya.

<p align="center"><img src = https://www.researchgate.net/publication/351542039/figure/fig1/AS:11431281172877200@1688685833363/Flow-diagram-of-gradient-boosting-machine-learning-method-The-ensemble-classifiers.png>

# **CONCLUSION & RECOMMENDATION**

## *RECOMMENDATION*
Strategi Bisnis yang Dapat Diterapkan
1. Tingkatkan Fasilitas Parkir: Mengingat ketersediaan parkir sangat berpengaruh, investasi dalam fasilitas parkir atau memberikan informasi yang jelas tentang parkir bisa mengurangi pembatalan.

2. Kebijakan untuk Riwayat Pembatalan: Terapkan kebijakan khusus bagi tamu dengan riwayat pembatalan, seperti deposit lebih tinggi atau pengingat reservasi tambahan.

3. Program Loyalitas untuk Pelanggan Transient: Buat program loyalitas atau penawaran khusus untuk meningkatkan komitmen tamu transient.

4. Optimalkan Manajemen Permintaan Khusus: Pastikan permintaan khusus tamu dapat dipenuhi atau komunikasikan batasan dengan jelas.

5. Kerjasama dengan Agen Perjalanan Online: Tinjau kebijakan pembatalan dan insentif bagi tamu yang memesan melalui agen perjalanan online untuk mengurangi pembatalan.

6. Analisis Negara Asal Tamu: Lakukan analisis lebih mendalam tentang mengapa tamu dari negara sendiri (Portugal) lebih sering membatalkan dan sesuaikan strategi pemasaran dan komunikasi.

Beberapa penambahan dan peningkatan yang disarankan untuk dilakukan agar dap meningkatkan performa model dan pengenmbangan bisnis :

- Menambahkan fitur yang dapat diisi oleh tamu terkait tingkat kepuasan dan pengalaman menginap agar dapat diketahui apakah tamu yang cancel disebabkan oleh kualitas layanan yang buruk atau tidak.
- Melakukan anonimasi nama tamu yang menginap sehingga tiap data menjadi unik serta tidak menimbulkan banyak nilai duplikat dan jumlah data yang dipelajari oleh model bisa jadi lebih banyak.
- Menambahkan fitur yang berisi sudah berapa kali tamu menginap di hotel tersebut, sehingga kita dapat melakukan pemetaan untuk menjaga loyalitas tamu dan mengetahui bagian apa dari hotel yang bisa menjadi daya tarik bagi tamu untuk datang kembali. 
- Mencoba metode handling outlier yang berbeda seperti Transform Method (Log Transformation, Square Root Transformation) atau dengan metode clustering seperti (DBSCAN)
- Menambahkan fitur-fitur atau kolom baru yang berisi durasi atau biaya penggunaan produk-produk yang ada seperti panggilan suara, SMS, dan internet. Sehingga perusahaan dapat melakukan segmentasi pelanggan untuk menentukan jenis produk yang paling sesuai untuk ditawarkan.
- Menganalisa data-data yang model yang masih salah prediksi (False Negative dan False Positive) untuk mengetahui alasan dan karakteristiknya.

## *CONCLUSSION*

- Metric yang kita gunakan adalah **recall score**, karena kita ingin agar model dapat mengurangi jumlah False Negative (pelanggan yang aktualnya cancel tetapi diprediksi No Cancel).
    <br>
    <br>
- Berdasarkan hyperparameter tuning, parameter terbaik yang dapat digunakan untuk benchmark model Decision Tree adalah :
    - n_estimators = 88 
    - max_features = 3
    - max_depth = 3
    - learning_rate = 0.08
    <br>
    <br>
- Berdasarkan permodelan Gradient Boost, feature `required_car_parking_space` adalah yang paling berpengaruh bagi tamu untk melakukan cancel atau tidak yang kemudian diikuti dengan `previous_cancellation` dan `customer_type`.
    <br>
    <br>
- Berdasarka hasil interpretasi pada Gradient Boost maka kita dapat simpulkan hal-hal sebagai berikut :
    - `required_car_parking_space` dan `previous_cancelation` sangat berpengaruh terhadap keputusan seorang tamu untuk melakukan cancel atau tidak
    - Tamu yang sebelumnya telah melakukan pembatalan pesanan lebih dari 10 kali maka memiliki kecenderungan besar untuk membatalkan pesanan
    - Pelanggan dengan tipe Transient (pelanggan yang hanya menginap selama satu malam/periode singkat) memiliki kecenderungan besar untuk membatalkan pesanan
    - Tamu yang berasal dari Portugal adalah tamu yang paling banyak membatalkan pesanan
    
    <br>
    <br>
- Berdasarkan contoh perhitungan biaya :
    - Kerugian hotel per cancel = 1,15x dari harga kamar per malam
    - Pelanggan yang melakukan cancel sebelum diterapklan machine learning = FN+TP = 328 orang
    - Total kerugian hotel sebelum machine learning = 856 orang x 1,15kali harga kamar = **377,2 kali harga kamar**
    - dalam USD = 377.2 x $99 = $37,342.8

    - Pelanggan yang gagal diprediksi cancel = 64 orang
    - Total kerugian yang gagal diantisipasi hotel = 64 orang x 1,15kali harga kamar = **73,6 kali harga kamar**
    - dalam USD = 73.6 x $99 = $7,286.4

    Dengan menggunakan machine learning hotel tersebut dapat meminimalkan kerugian hingga **80,49%** atau sebesar **$30,056.4**
    <br>
    <br>





