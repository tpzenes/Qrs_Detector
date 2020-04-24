import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.signal import butter, lfilter
from scipy import signal
from scipy import fftpack

#Gürültü üreten fonksiyon
def noise_uret():
    import random
    sample = 12800
    x = np.arange(sample)
    pre_noise = np.asarray(random.sample(range(0,100000),sample))
    noise=[]
    for i in range(len(pre_noise)):
        noise.append(np.random.choice([0.000007,-0.000007])*pre_noise[i])
    return noise

def band_geciren_filtre(ekg_verisi, altkesim, ustkesim, ornekleme_frekansı, filtre_sirasi):
        nyquist_freq = 0.5 * ornekleme_frekansı
        alt = altkesim / nyquist_freq
        ust = ustkesim / nyquist_freq
        b, a = butter(filtre_sirasi, [alt, ust], btype="band")
        y = lfilter(b, a, ekg_verisi)
        return b,a,y

#Tepe değerleri bulan fonksiyon
def qrs_belirle(peak_degerleri,carpilmis_ekg):
    peak_degerleri = list(peak_degerleri)
    i=0
    yeni_veriler = []
    while i < len(carpilmis_ekg):
        try:
            if carpilmis_ekg[i+10] in peak_degerleri:
                peak = peak_degerleri[0]
                if carpilmis_ekg[i+10] != peak_degerleri[0]:
                    peak_degerleri.pop(0)
                while carpilmis_ekg[i-10] not in peak_degerleri:
                    yeni_veriler.append(peak)
                    i+=1
            elif carpilmis_ekg[i-10] in peak_degerleri and i>=10:
                peak_degerleri.pop(0)
                while carpilmis_ekg[i+10] not in peak_degerleri:
                    yeni_veriler.append(0)
                    i+=1
            else:
                yeni_veriler.append(0)
                i+=1
        except:
            break
    return yeni_veriler
    
#Tepe değerleri bulan fonksiyon
def peak_bul(veri, peak_limit, peak_araligi):
    len = veri.size
    x = np.zeros(len + 2 * peak_araligi)
    x[:peak_araligi] = veri[0] - 1.e-6
    x[-peak_araligi:] = veri[-1] - 1.e-6
    x[peak_araligi:peak_araligi + len] = veri
    peak_adayi = np.zeros(len)
    peak_adayi[:] = True
    for i in range(peak_araligi):
        baslangic = peak_araligi - i - 1
        h_b = x[baslangic: baslangic + len]  # önce
        baslangic = peak_araligi
        h_c = x[baslangic: baslangic + len]  # merkez
        baslangic = peak_araligi + i + 1
        h_a = x[baslangic: baslangic + len]  # sonra
        peak_adayi = np.logical_and(peak_adayi, np.logical_and(h_c > h_b, h_c > h_a))
    ind = np.argwhere(peak_adayi)
    ind = ind.reshape(ind.size)
    ind = ind[veri[ind] > peak_limit]
    return ind    
def main():
    filtre_sirasi = 6
    ekg_path = "C:/Users/enes.topuz/Downloads/Compressed/Odev1_BD_2020/N_type_beats.txt"
    altkesim = 5                   #Band geçiren filterinin alt kesim frekansı
    ustkesim = 15                 #Band geçiren filterinin üst kesim frekansı
    ornekleme_frekansı = 360      # EKG örnekleme frekansı

    ekg_olcumleri = np.loadtxt(ekg_path, delimiter='\n')  #datayı yükle
    noise = noise_uret()
    #ekg_olcumleri += noise   #Gürültü eklenen kısım (Devre dışı bırakmak için yorum haline getiriniz.)
    b,a,yfiltrelenmis_ekg = band_geciren_filtre(ekg_olcumleri,altkesim,ustkesim,ornekleme_frekansı,filtre_sirasi)  #ham data band geçiren filtreden geçirilerek filtreleniyor.
    turev_alinmis_ekg_olcumleri = np.ediff1d(filtrelenmis_ekg)         #turev alınıyor
    turevin_karesi = turev_alinmis_ekg_olcumleri ** 2  #karesi alınıyor
    integrasyon_penceresi = 54
    integre_ekg_verileri = np.convolve(turevin_karesi, np.ones(integrasyon_penceresi))  #Hareket eden pencere ile integrasyon
    
    geciktirilmis_ekg = list(ekg_olcumleri[:])
    for i in range(34):
        geciktirilmis_ekg.insert(0,ekg_olcumleri[i])
    carpilmis_ekg = np.multiply(integre_ekg_verileri[:int(len(integre_ekg_verileri)/4)],geciktirilmis_ekg[:int(len(geciktirilmis_ekg)/4+5)])
    
    peak_limit = 0.1   # peak değer limiti
    peak_araligi = 25
    
    # Çarpılmış EKG için Peak Değerleri
    peak_indisleri = peak_bul(carpilmis_ekg,peak_limit,peak_araligi)
    peak_degerleri = carpilmis_ekg[peak_indisleri]
    
    #Ham EKG Ölçümleri için Peak Değerleri
    peak_indisleri_2 = peak_bul(ekg_olcumleri,peak_limit,peak_araligi)
    peak_degerleri_2 = ekg_olcumleri[peak_indisleri_2]    
    
    #Qrs Deteksiyonu
    yeni_veriler = qrs_belirle(peak_degerleri,carpilmis_ekg)
    
    #R-R süresi hesaplama
    #Algoritma 43 adet indis bulması gerekirken 42 adet buluyor.(1 tane kaçırıyor). Bu yüzden 42 adet R-R süresi yerine 40 adet RR süresi bulunuyor.
    diff=[]
    for i in range(1,len(peak_indisleri_2[:19])):
        diff.append(peak_indisleri_2[i] - peak_indisleri_2[i-1])
    for i in range(20,len(peak_indisleri_2)):
        diff.append(peak_indisleri_2[i] - peak_indisleri_2[i-1])
    
    
    plt.figure(figsize=(30,8))
    plt.title("Ham EKG Olcumleri--040150057",fontsize="30")
    plt.ylabel("Genlik (V)",fontsize="20")
    plt.xlabel("Zaman (ms)",fontsize="20")
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(ekg_olcumleri[:int(len(ekg_olcumleri)/4)])
    
    plt.figure(figsize=(30,8))
    plt.title("Band Geçiren Filtre Frekans Cevabı--040150057",fontsize=30)
    plt.xlabel("Frekans (Hz)",fontsize=20)
    plt.ylabel("Genlik (V)",fontsize=20)
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(((ornekleme_frekansı * 0.5 / np.pi) * w)[:300], abs(h)[:300])
    
    plt.figure(figsize=(30,8))
    plt.title("Filtrelenmiş EKG--040150057",fontsize="30")
    plt.ylabel("Genlik (V)",fontsize="20")
    plt.xlabel("Zaman (ms)",fontsize="20")
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(filtrelenmis_ekg[:int(len(filtrelenmis_ekg)/4)])
    
    plt.figure(figsize=(30,8))
    plt.title("Turev Alinmis EKG--040150057",fontsize="30")
    plt.ylabel("Genlik (V)",fontsize="20")
    plt.xlabel("Zaman (ms)",fontsize="20")
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(turev_alinmis_ekg_olcumleri[:int(len(turev_alinmis_ekg_olcumleri)/4)])
    
    plt.figure(figsize=(30,8))
    plt.title("Turev'in Karesi EKG--040150057",fontsize="30")
    plt.ylabel("Genlik (V)",fontsize="20")
    plt.xlabel("Zaman (ms)",fontsize="20")
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(turevin_karesi[:int(len(turevin_karesi)/4)])
    
    plt.figure(figsize=(30,8))
    plt.title("Integre Edilmis EKG--040150057",fontsize="30")
    plt.ylabel("Genlik (V)",fontsize="20")
    plt.xlabel("Zaman (ms)",fontsize="20")
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(integre_ekg_verileri[:int(len(integre_ekg_verileri)/4)])
    
    plt.figure(figsize=(30,8))
    plt.title("Çarpılmış EKG, Peak değerler işaretli--040150057",fontsize="30")
    plt.ylabel("Genlik (V)",fontsize="20")
    plt.xlabel("Zaman (ms)",fontsize="20")
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(carpilmis_ekg)
    plt.scatter(peak_indisleri,peak_degerleri,marker="o",color="black")
    
    plt.figure(figsize=(30,8))
    plt.title("Ham EKG Ölçümleri, Peak değerler işaretli--040150057",fontsize="30")
    plt.ylabel("Genlik (V)",fontsize="20")
    plt.xlabel("Zaman (ms)",fontsize="20")
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(ekg_olcumleri)
    plt.scatter(peak_indisleri_2,peak_degerleri_2,marker="o",color="black")
    
    plt.figure(figsize=(30,8))
    plt.title("İntegre ve Geciktirilmiş EKG--040150057",fontsize="30")
    plt.ylabel("Genlik (V)",fontsize="20")
    plt.xlabel("Zaman (ms)",fontsize="20")
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(geciktirilmis_ekg[:int(len(geciktirilmis_ekg)/4)],label="Geciktirilmis EKG")
    plt.plot(integre_ekg_verileri[:int(len(integre_ekg_verileri)/4)],label = "İntegre Edilmiş EKG")
    plt.legend(fontsize="20",loc="upper right")

    
    plt.figure(figsize=(30,8))
    plt.title("İntegre ve Geciktirilmiş EKG Çarpımı--040150057",fontsize="30")
    plt.ylabel("Genlik (V)",fontsize="20")
    plt.xlabel("Zaman (ms)",fontsize="20")
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(carpilmis_ekg)
    
    
    plt.figure(figsize=(30,8))
    plt.title("Darbe ve Geciktirilmiş EKG--040150057",fontsize="30")
    plt.ylabel("Genlik (V)",fontsize="20")
    plt.xlabel("Zaman (ms)",fontsize="20")
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(geciktirilmis_ekg[:int(len(geciktirilmis_ekg)/4)],label="Geciktirilmis EKG")
    plt.plot(yeni_veriler,label="Darbe")
    plt.legend(fontsize="25",loc="upper right")
    
    plt.figure(figsize=(30,8))
    plt.title("Darbe ile QRS Deteksiyonu--040150057",fontsize="30")
    plt.ylabel("Genlik (V)",fontsize="20")
    plt.xlabel("Zaman (ms)",fontsize="20")
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(yeni_veriler)
    plt.plot(carpilmis_ekg)
    
    #Ham EKG Spektrum
    X = fftpack.fft(ekg_olcumleri)
    freqs = fftpack.fftfreq(len(ekg_olcumleri)) * ornekleme_frekansı
    
    fig, ax = plt.subplots(figsize=(30,8))
    
    ax.stem(freqs, np.abs(X),use_line_collection=True)
    ax.tick_params(labelsize=15,labelcolor="red")
    ax.set_xlabel('Frekans (Hz)',fontsize="20")
    ax.set_ylabel('Genlik',fontsize="20")
    ax.set_title("Ham EKG Ölçümleri Frekans Spektrumu--040150057",fontsize="30")
    ax.set_xlim(-ornekleme_frekansı / 2, ornekleme_frekansı / 2)
    
    #Filtrelenmis EKG Spektrum
    X = fftpack.fft(filtrelenmis_ekg)
    freqs = fftpack.fftfreq(len(filtrelenmis_ekg)) * ornekleme_frekansı
    
    fig, ax = plt.subplots(figsize=(30,8))
    
    ax.stem(freqs, np.abs(X),use_line_collection=True)
    ax.tick_params(labelsize=15,labelcolor="red")
    ax.set_xlabel('Frekans [Hz]',fontsize="30")
    ax.set_ylabel('Genlik',fontsize="30")
    ax.set_title("Filtrelenmiş EKG Ölçümleri Frekans Spektrumu--040150057",fontsize="30")
    ax.set_xlim(-ornekleme_frekansı / 2, ornekleme_frekansı / 2)
    
    plt.figure(figsize=(30,8))
    plt.title("R-R Arası Zaman Farkı--040150057",fontsize="30")
    plt.ylabel("Fark (ms)",fontsize="20")
    plt.xlabel("İndisler",fontsize="20")
    plt.tick_params(labelsize=15,labelcolor="red")
    plt.plot(diff,marker='o')
    
    return integre_ekg_verileri,peak_indisleri,peak_degerleri,ekg_olcumleri,filtrelenmis_ekg,ornekleme_frekansı,\
    carpilmis_ekg,peak_degerleri,peak_degerleri_2,peak_indisleri_2,filtre_sirasi,altkesim,ustkesim

    
    
integre_ekg_verileri,peak_indisleri,peak_degerleri,ekg_olcumleri,\
filtrelenmis_ekg,ornekleme_frekansı,carpilmis_ekg,peak_degerleri,peak_degerleri_2,peak_indisleri_2,filtre_sirasi,\
altkesim,ustkesim = main()
