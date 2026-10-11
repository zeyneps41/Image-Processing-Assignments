import cv2
import numpy as np
import matplotlib.pyplot as plt


image_bgr = cv2.imread('1.png')


if image_bgr is None:
    print("Görüntü bulunamadı! Lütfen dosya yolunu kontrol edin.")
else:
    
    print("--- GÖRÜNTÜ ÖZELLİKLERİ ---")
    print(f"Shape (Boyutlar): {image_bgr.shape}")
    print(f"Dtype (Veri Tipi): {image_bgr.dtype}")
    print(f"Min Piksel Değeri: {image_bgr.min()}")
    print(f"Max Piksel Değeri: {image_bgr.max()}")
    print(f"Nbytes (Bellekte Kapladığı Alan): {image_bgr.nbytes} byte\n")

   
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

   
    ornek_piksel = image_rgb[150, 200]
    print(f"(150, 200) koordinatındaki RGB Piksel Değeri: {ornek_piksel}")

    
    roi = image_rgb[100:400, 200:500] 

    

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

   
    image_negative = 255 - image_rgb

    image_bright = cv2.add(image_rgb, np.array([50.0]))

    _, image_thresh = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)


    
    plt.figure(figsize=(15, 10))

    
    plt.subplot(3, 3, 1)
    plt.imshow(image_bgr)
    plt.title('1. BGR Görüntü (Yanlış Renk)')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(image_rgb)
    plt.title('2. RGB Görüntü (Doğru Renk)')
    plt.axis('off')

   
    plt.subplot(3, 3, 3)
    plt.imshow(roi)
    plt.title('3. Seçilen ROI (Bölge)')
    plt.axis('off')

    
    plt.subplot(3, 3, 4)
    plt.imshow(image_gray, cmap='gray')
    plt.title('4. Gri Görüntü')
    plt.axis('off')

    
    plt.subplot(3, 3, 5)
    plt.imshow(image_negative)
    plt.title('5. Negatif Görüntü')
    plt.axis('off')

   
    plt.subplot(3, 3, 6)
    plt.imshow(image_bright)
    plt.title('6. +50 Parlaklık')
    plt.axis('off')

    
    plt.subplot(3, 3, 7)
    plt.imshow(image_thresh, cmap='gray')
    plt.title('7. Threshold (T=128)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



""" 
Çıktıların Yorumlanması:

Shape: Çıktı genellikle (Yükseklik, Genişlik, 3) şeklinde döner. Buradaki 3, görüntünün 3 kanallı (Renkli - BGR/RGB) olduğunu belirtir.

Dtype: Piksellerin "Unsigned Integer 8-bit" formatında olduğunu gösterir. Yani her bir piksel kanalı 0 ile 255 arasında tam sayı değeri alabilir.

Min ve Max: Görüntüdeki en karanlık (0'a yakın) ve en parlak (255'e yakın) piksel değerlerini gösterir.

Nbytes: Görüntünün RAM üzerinde kapladığı toplam bayt miktarını verir. (Yükseklik * Genişlik * Kanal Sayısı * 1 bayt formülü ile hesaplanır).

Neden image[y, x] Kullanılır?
Görüntüler bilgisayar belleğinde çok boyutlu matrisler (diziler) olarak tutulur. Matematikte ve lineer cebirde bir matrisin elemanlarına erişirken önce satır (row), sonra sütun (column) indeksi belirtilir; yani matris[satır, sütun].
Görüntü işleme dünyasında:

Satırlar (Rows): Görüntünün yukarıdan aşağıya doğru Y eksenini (yüksekliğini) temsil eder.

Sütunlar (Columns): Görüntünün soldan sağa doğru X eksenini (genişliğini) temsil eder.
Bu matris mantığından dolayı, bir piksele erişmek için önce yataydaki satır indeksi (y), ardından dikeydeki sütun indeksi (x) yazılır: image[y, x].

BGR / RGB Farkı Neden Oluşur?
OpenCV kütüphanesi, tarihsel nedenlerden ötürü pikselleri okurken Renk Kanallarını Kırmızı-Yeşil-Mavi (RGB) yerine Mavi-Yeşil-Kırmızı (BGR) sırasıyla dizer. Ancak Matplotlib gibi çizim kütüphaneleri ve standart ekranlar RGB sırasını bekler. Eğer cv2.imread ile okunan görüntüyü çevirmeden doğrudan Matplotlib'e verirsek, Kırmızı ve Mavi pikseller yer değiştireceği için muz mavi, mavi bardak ise kırmızı tonlarında görünecektir (kodda üretilen ilk 2 görsel bunu kanıtlar).
"""