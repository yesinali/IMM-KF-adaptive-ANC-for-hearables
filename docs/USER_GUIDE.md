# Kullanım Kılavuzu — IMM-KF Adaptive ANC Demo

Bu kılavuz, projenin **interaktif demosunu** sıfırdan kendi bilgisayarında çalıştırmak için her şeyi anlatır. Programlama bilgin olmasa bile takip edebilmen için adım adım yazılmıştır.

---

## 1. Bu uygulama ne yapıyor?

Kısaca: **Bir kulaklığın gürültü engelleme (ANC) algoritmasının PC üzerinde simülasyonu.**

Kulaklığın çevresi sürekli değişir (sessiz oda → trafik → rüzgar → kalabalık). Klasik tek-ayarlı filtreler bu değişimlerde ya çok hassas kalıp tepki veremez ya çok agresif olup gereksiz gürültü ekler. Bu projede **IMM-KF** denilen Bayesian bir yöntem var — 4 farklı ayarı paralel çalıştırıp **otomatik karıştıran** akıllı bir filtre. Demoda bu algoritmayı klasik yöntemlerle karşılaştırıp **kulağınla farkı duyabiliyorsun.**

Demoda yapabilecekler:
- Sentetik gürültü senaryosu üret (sessiz/trafik/rüzgar/konuşma)
- Veya kendi WAV ses dosyanı yükle
- 3 farklı algoritmadan birini seç (NLMS / Kalman / IMM)
- Ayarları slider'larla değiştir
- Orijinal gürültüyü ve ANC sonrası kalan sesi **yan yana dinle**
- Grafiklerde performansı incele

---

## 2. Sistem gereksinimleri

| Gereksinim | Detay |
|---|---|
| **İşletim sistemi** | Windows 10/11, macOS, veya Linux (hepsi çalışır) |
| **Python** | 3.10 veya üstü |
| **Disk** | ~500 MB (Python kütüphaneleri dahil) |
| **RAM** | 4 GB yeter |
| **Tarayıcı** | Chrome, Edge, Firefox veya Safari (güncel sürüm) |
| **İnternet** | Sadece kurulum sırasında gerekli (paketleri indirmek için) |

**Önemli:** Sesleri dinleyebilmek için hoparlör veya kulaklık olmalı.

---

## 3. Kurulum (ilk seferlik, ~5 dakika)

### Adım 3.1 — Python'un kurulu olup olmadığını kontrol et

PowerShell veya CMD aç (Windows tuşu → "powershell" yaz → enter), şunu yaz:

```powershell
python --version
```

**Eğer "Python 3.10.x" veya üstü görüyorsan:** geç Adım 3.3'e.

**Eğer "command not found" veya hiç bir şey görmüyorsan:** Adım 3.2'ye git.

### Adım 3.2 — Python kur (eğer yoksa)

1. https://www.python.org/downloads/ adresine git
2. Sarı **"Download Python 3.12"** butonuna bas (veya en güncel sürüm)
3. İndirilen `.exe` dosyasını çalıştır
4. **ÇOK ÖNEMLİ:** İlk ekrandaki **"Add Python to PATH"** kutucuğunu işaretle (yoksa sonradan komutlar çalışmaz)
5. "Install Now" tuşuna bas, kurulumu bekle
6. Bittiğinde PowerShell'i kapat ve yeniden aç (PATH değişikliklerinin geçerli olması için)
7. Adım 3.1'i tekrar yap, artık Python görmeli

### Adım 3.3 — Proje klasörünü al

Projeyi sana gönderene göre:
- **ZIP olarak aldıysan:** Bir yere çıkar (örn. `Belgeler\kalman_proje\`)
- **GitHub'dan klonladıysan:** `git clone <URL>` ile aldığın klasörü kullan

Klasörün içinde `src/`, `scripts/`, `app/` gibi alt klasörler olmalı.

### Adım 3.4 — Proje klasörüne git (terminalde)

PowerShell'de proje klasörüne git:

```powershell
cd "C:\Users\KULLANICI_ADIN\Belgeler\kalman_proje"
```

(Yukarıdaki yolu kendi durumuna göre değiştir. `cd` komutu "klasöre git" demek.)

İçinde olduğunu doğrulamak için:

```powershell
ls
```

Şöyle bir şeyler görmeli: `src`, `scripts`, `app`, `requirements.txt`, `README.md`...

### Adım 3.5 — Gerekli paketleri yükle

```powershell
pip install -r requirements.txt
```

Bu komut yaklaşık **1-3 dakika** sürer. Şunları yükler:
- NumPy (matematiksel hesaplamalar)
- SciPy (sinyal işleme)
- Matplotlib (grafikler)
- Soundfile (ses dosyaları)
- Streamlit (arayüz)

Sona "Successfully installed..." yazısı gelirse tamamdır.

**Hata alırsan:** Adım 7 (Sorun Giderme) bölümüne bak.

---

## 4. Demoyu çalıştırma

### Adım 4.1 — Streamlit'i başlat

Hâlâ proje klasöründeyken PowerShell'de:

```powershell
streamlit run app/streamlit_app.py
```

İlk çalıştırırken e-mail soracak — direkt **enter**'a basıp geçebilirsin (zorunlu değil).

Sonra terminalde şöyle bir mesaj göreceksin:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Adım 4.2 — Tarayıcıyı aç

**Otomatik açılırsa:** harika, hazırsın.

**Otomatik açılmazsa:** Chrome / Edge / Firefox aç ve adres çubuğuna **`http://localhost:8501`** yaz, enter'a bas.

Karşına demo arayüzü çıkacak.

### Adım 4.3 — Demoyu kapatmak

Terminal penceresinde **Ctrl + C** tuşlarına bas. Sunucu kapanır.

---

## 5. Arayüz rehberi

Arayüzde 2 ana alan var:

### 5.1 Sol panel (sidebar) — ayarlar

#### 🚀 Demo presets (en üstte — kestirme)
Üç hazır buton: **IMM-KF (v5)**, **Quiet-Kalman trap**, **NLMS baseline**. Birine basınca
algoritma + parametreler otomatik kurulur ve simülasyon **kendiliğinden başlar** — aşağıdaki
ayarlarla tek tek uğraşmak istemiyorsan buradan başla.

#### Audio source (ses kaynağı)
- **Synthetic:** Hazır sentetik senaryo
- **Upload WAV:** Kendi ses dosyanı yükle

#### Synthetic seçildiğinde:
- **Quiet / Traffic / Wind / Babble sliders:** Her gürültü tipinin süresi (saniye). Mesela quiet=5, traffic=10 dersen önce 5 saniye sessiz, sonra 10 saniye trafik gürültüsü olur.
- **Mode-conditioned plants:** İşaretli bırak — daha zorlu (dinamik) bir test. Bu seçenek her gürültü tipinde kulaklığın akustik ortamının da değiştiğini simüle eder.
- **Random seed:** Aynı rastgele test için aynı sayıyı kullan. Farklı senaryolar için değiştir (1-9999 arası).

#### Upload WAV seçildiğinde:
- WAV dosyanı sürükle veya "Browse files" ile seç. Otomatik olarak 16 kHz'e dönüştürülür.

#### Method (algoritma) — 3 seçenek:
- **NLMS:** Klasik adaptif filtre, endüstri standardı
- **Kalman (single mode):** Tek-modlu Bayesian filtre, sabit ayarlı
- **IMM-KF (4 modes):** Bu projenin önerdiği akıllı Bayesian filtre

#### Filter length L:
FIR filtre uzunluğu. **64'te bırak** — değiştirmeye gerek yok.

#### Yönteme göre değişen ayar:
- **NLMS seçtiysen:** `µ` (step size, adaptasyon hızı). 0.10 dene başlangıçta.
- **Kalman seçtiysen:** `log10(σ_q²)` ve `log10(σ_r²)` — bu Q ve R değerlerinin 10 üssü cinsinden ifadesi. Yavaş Kalman için: Q=-12, R=0. Hızlı Kalman için: Q=-5, R=2.
- **IMM seçtiysen:** `Likelihood window` — mod kararının zaman pencereli yumuşatması. **200'de bırak.**

#### Compute backend
- **Python NumPy:** Varsayılan; IMM'in mod-posteriörü grafiği için gerekli.
- **Pure C / OpenBLAS:** C portu derlenmişse görünür (bkz. README, kurulum bölümü).
  ~9 kat hızlı, sonuç bire bir aynı.

#### Run butonu
Mavi büyük tuş — ayarları elle kurduysan basacaksın (preset kullandıysan gerek yok).

### 5.2 Ana panel — sonuçlar

Run bittikten sonra (algoritma ve backend'e göre 5 saniye – 3 dakika sürer):

#### Run seçici ve geçmiş
Aynı oturumdaki **her koşu hafızada tutulur** (en fazla 8). Sonuçların üstündeki
**"📂 Showing results of run"** menüsünden eski bir koşuyu seçersen tüm kartlar,
sesler ve grafikler **anında** o koşuya döner — yeniden hesaplama yok. Altta açılan
**"📚 Compare with previous runs"** tablosunda koşular yan yana (her satırda kalıntının
ses oynatıcısı da var).

#### Üst kısım: 2 sıra KPI kartı
- **Overall NR:** Toplam gürültü azalması (dB). **Yüksek olsun.**
- **Audio length / Mode tracking / backend hızı (µs-sample, RTF):** koşu bilgileri
- İkinci sıra **algı metrikleri:** algılanan ses düşüşü dB(A), alçak/yüksek bant NR,
  müziksel-gürültü indeksi (düşük = daha doğal kalıntı)

#### Audio comparison (ses karşılaştırma)
**İki ses oynatıcı:**
- **Solda:** Orijinal gürültü d(k) — ANC kapalıyken kulağına gelen
- **Sağda:** ANC sonrası kalan e(k) — ANC açıkken kulağına gelen

**Sırayla ikisini de dinle**, farkı kulağınla duyuyor musun? Bu demonun en güzel kısmı.
(Sidebar'daki "virtual headphone" kutusunu işaretlersen, seviye farkını koruyan
kulaklık-simülasyonu sürümünü dinlersin.)

#### Visualization (görselleştirme) — 6 sekme:
- **📈 Time domain:** Dalga şekilleri (üstte orijinal, altta ANC sonrası). Genliğin azalması beklenir.
- **🌈 Spectrogram:** Önce/sonra spektrogramları, ortak renk skalası — ANC açılınca alçak bant kararır.
- **📉 NR over time:** Zaman içinde NR'nin değişimi (dB grafiği). Yüksek = iyi.
- **🎯 Mode posteriors:** Sadece IMM'de anlamlı. IMM'in her an hangi modu seçtiğini gösterir.
- **🎚️ NR per frequency:** 1/3-oktav bantlarda azaltım — aktif ANC alçak frekanslarda çalışır.
- **🆚 Overlay runs:** Geçmişteki koşuların NR eğrileri **tek eksende üst üste** — IMM ile
  sabit filtre arasındaki farkı tek grafikte görmenin en hızlı yolu.

---

## 6. Önerilen denemeler

Sırayla şunları dene, sonuçları karşılaştır:

### Deney 1 — IMM'in temel performansı
1. Synthetic seç, default ayarları kullan (her mod 5 saniye)
2. Mode-conditioned plants **işaretli kalsın**
3. Method: **IMM-KF**
4. Run'a bas (~30 saniye sürer)
5. **Ses A/B'sini dinle, NR'a bak**
6. Mode posteriors sekmesini aç — IMM modları nasıl tanıdı?

### Deney 2 — IMM vs klasik NLMS
1. Aynı senaryo, aynı seed (örn. 7)
2. Method: **NLMS**, µ = 0.10
3. Run, sonra NR ve sesi karşılaştır
4. IMM'in NR'ı klasik yöntemden büyük mü? **Beklenen evet.**

### Deney 3 — Yavaş Kalman çöküşü (projenin asıl bulgusu)
1. Aynı senaryo (mode-conditioned plants **açık** kalsın — bu kritik)
2. Sidebar'ın en üstündeki **🪤 Quiet-Kalman trap** preset'ine bas (elle kurmak istersen:
   Kalman (single mode), `log_q = -12`, `log_r = 2`)
3. Beklenti: **NR çok düşük** (örn. +2 dB), çünkü yavaş filtre ortam değişimlerine adapte olamıyor
4. Sesi dinle — gürültü neredeyse hiç azalmamış gibi gelecek
5. **🆚 Overlay runs** sekmesini aç: Deney 1'deki IMM eğrisiyle bu koşu aynı eksende —
   fark tek bakışta görünür

Bu deney projenin asıl tezini doğruluyor: "tek bir sabit ayar, çevre değişen ortamda yetersizdir."

### Deney 4 — Kendi sesinle test
1. Telefonla 5-10 saniyelik bir gürültü kaydı yap (trafik, kafe, vb.)
2. WAV formatında kaydet, bilgisayara aktar
3. Demoda "Upload WAV" seç, dosyayı yükle
4. IMM seç, Run
5. Kendi gürültünde algoritma nasıl çalıştı?

---

## 7. Sorun giderme

### "streamlit: command not found"
Streamlit kurulu ama PATH'te değil. Şunu dene:
```powershell
python -m streamlit run app/streamlit_app.py
```

### "ModuleNotFoundError: No module named 'XYZ'"
Paketler düzgün yüklenmemiş. Tekrar yükle:
```powershell
pip install -r requirements.txt --upgrade
```

### "pip: command not found"
Python kurulumun eski veya bozuk. Adım 3.2'yi yeniden yap, **"Add Python to PATH"** kutucuğunun işaretli olduğundan emin ol.

### Streamlit açıldı ama "Network error" gösteriyor
Anti-virüs veya güvenlik duvarı engelliyor olabilir. Geçici olarak kapatıp dene. Çalışırsa exception kuralı ekle.

### Tarayıcıda açıldı ama Run'a basınca hata
Terminale dön — orada Python hatası görürsen, bana göster.

### Demo açıldı ama çok yavaş çalışıyor
IMM-KF Python'da yavaş. Daha hızlı sonuç için:
- Senaryo sürelerini kısalt (her segment için 3-5 sn yeter)
- Veya NLMS / Kalman seç (çok daha hızlı)

### Ses çalmıyor / dinleyemiyorum
- Tarayıcıdaki ses oynatıcısının üzerine tıklayıp play tuşuna bas
- Tarayıcı izin istiyorsa "izin ver" de
- Sistem ses çıkışını kontrol et

### Aynı port hatası (port 8501 already in use)
Önceki Streamlit sunucusu hâlâ açık. Terminale dön ve Ctrl+C bas. Veya:
```powershell
streamlit run app/streamlit_app.py --server.port 8502
```
Sonra tarayıcıda `http://localhost:8502` aç.

---

## 8. Bonus: Komut satırından scriptleri çalıştırmak

Demo dışında, projeyi komut satırından çalıştırmak istersen:

```powershell
python -m scripts.01_inspect_paths       # Akustik yol görselleri
python -m scripts.02_inspect_scenario    # Senaryo görselleştirme + WAV
python -m scripts.03_baseline_run        # Statik baseline karşılaştırma
python -m scripts.05_dynamic_imm         # Dinamik IMM testi
python -m scripts.06_monte_carlo --runs 5  # Monte Carlo (5 run, ~10 dk)
```

Çıktılar `figures/` klasörüne kaydedilir.

---

## 9. Hızlı referans

| Ne istiyorsun? | Komut / İşlem |
|---|---|
| Python var mı kontrol | `python --version` |
| Paketleri yükle | `pip install -r requirements.txt` |
| Demoyu başlat | `streamlit run app/streamlit_app.py` |
| Demoyu durdur | Terminalde Ctrl+C |
| Tarayıcıda aç | `http://localhost:8501` |
| Demo kapanmadan tarayıcıyı yenile | Tarayıcıda F5 |

---

## İletişim

Projenin teknik dokümantasyonu için [README.md](README.md) ve final raporu (`EE4084_Final_Report.pdf`) bak.

Demoyu çalıştırırken takıldığın bir adım varsa, hangi adımda ne hatası aldığını not et — terminal çıktısının tamamı çok yardımcı olur.

İyi denemeler!
