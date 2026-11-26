# Teacher utterance: Transformer embeddings + greedy discriminative parsing

## Hyperparameter search

`embedding_models.csv` berisi daftar model embedding dan batch size encoding.
`hyperparameter_search.csv` berisi 19 konfigurasi loss. Kolom `class_weight`
menerima `none` atau `balanced`; bobot seimbang dihitung hanya dari label pada
bagian training.

Jalankan skrip dari direktori ini:

```text
pip install -r requirements.txt
python teacher_intent_experiment.py
```

Skrip meng-encode dataset sekali untuk setiap embedding model dan menguji semua
19 konfigurasi loss dengan split train/validation/test yang sama. Dengan daftar
default, ada 4 x 19 kombinasi. Konfigurasi global dipilih berdasarkan macro-F1
validation (lalu parse exact match sebagai pemecah seri), dan test set hanya
dievaluasi untuk pilihan final. Untuk mengulang atau membatasi search, edit
baris di `embedding_models.csv` dan/atau `hyperparameter_search.csv`.

Training untuk konfigurasi loss pada satu embedding model dijalankan paralel
menggunakan proses CPU, dibatasi oleh jumlah konfigurasi dan CPU yang tersedia.
Thread internal library numerik dibatasi satu per worker untuk mencegah
oversubscription. Encoding embedding tetap menggunakan GPU jika CUDA tersedia.

Hasil validation tiap trial langsung disimpan ke
`outputs/hyperparameter_results.csv` saat trial selesai, sehingga tetap tersedia
jika proses berhenti sebelum seluruh search selesai. Kolom metrik test baru
diisi setelah konfigurasi terbaik dipilih dan test set dievaluasi.

`metrics.json`, `model.joblib`, `predictions.csv`, dan kurva training
menggambarkan model embedding serta konfigurasi loss terpilih. Model diproses
satu per satu; `batch_size` dapat diturunkan bila VRAM tidak mencukupi.

## Clause embedding

Encoder yang dibandingkan tercantum di `embedding_models.csv`.

- Pretrained Transformer sentence encoder.
- Dimensi embedding bergantung pada model yang dipilih.
- Encoder dibekukan (*frozen*), tidak di-fine-tune.
- Semua clause di-encode secara batch untuk menghindari pemanggilan model
  berulang kali per utterance.
- Pada penggunaan pertama, model akan diunduh jika belum tersedia di cache.

## File keluaran

- `outputs/model.joblib`: parameter model, nama Transformer terpilih,
  hyperparameter terpilih, dan metadata optimizer.
- `outputs/metrics.json`: metrik validasi dan test.
- `outputs/predictions.csv`: gold tree dan predicted tree untuk setiap utterance test.
- `outputs/training_history.csv`: nilai objective selama iterasi L-BFGS.
- `outputs/hyperparameter_results.csv`: metrik validation per konfigurasi dan
  metrik test hanya untuk konfigurasi terpilih.
- `outputs/training_curve.png`: kurva objective.
- `outputs/confusion_matrix.png`: confusion matrix intent.
- `outputs/class_f1.png`: F1 per kelas.

## Catatan metodologis

1. Leaf pada gold tree dipakai sebagai clause segmentation. Struktur internal gold
   hanya dipakai sebagai supervisi training dan evaluasi, bukan sebagai input decoder.
2. Clause embedding menggunakan pretrained Transformer Sentence Encoder
   dari `embedding_models.csv` dan tidak di-fine-tune.
3. `unlabeled` diperlakukan sebagai kelas kelima. Satu label gabungan yang tidak
   termasuk lima kelas diabaikan untuk supervised classification.
4. Semua pasangan adjacent dianggap legal untuk digabung karena dataset tidak
   menyediakan grammar constraint.
5. Subgradient menghitung gradien pada tree greedy dan kelas argmax yang sedang
   terpilih, tanpa mendiferensiasikan keputusan diskrit greedy/argmax.
6. L-BFGS-B dengan subgradient adalah pendekatan eksperimen untuk objective
   piecewise-differentiable; konvergensi tidak dijamin seperti pada objective halus.
7. Intent classification metrics dihitung pada node-node gold tree. Parsing metrics
   dihitung terpisah pada utterance dengan lebih dari satu clause.
