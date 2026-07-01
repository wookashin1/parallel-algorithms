import math
import os
from PIL import Image
import numpy as np
from functools import reduce


def load_images(data_set_path):
    images = []

    for cls in os.listdir(data_set_path):
        class_path = os.path.join(data_set_path, cls)

        #proveravam da li je class_path stvarno dir (zbog .DS_Store)
        if not os.path.isdir(class_path):
            continue


        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append((cls, img_path))

    return images


def list_length(lista):
    g = reduce(lambda x, _: x + 1 , lista, 0)
    return g

def custom_range(limit):
    return reduce(lambda acc, _: acc + [list_length(acc)], [None] * limit, [])

NUM_BINS = 8
def calculate_normalized_bins_histograms(image_path):# RADIII
    image = Image.open(image_path)

    image = image.convert('RGB')

    width, height = image.size

    bin_size = 256 // NUM_BINS
    initial_histograms = (
        [0.0] * NUM_BINS,
        [0.0] * NUM_BINS,
        [0.0] * NUM_BINS
    )

    def update_histogram(histograms, pixel):
        r, g, b = pixel
        r_bin = r // bin_size
        g_bin = g // bin_size
        b_bin = b // bin_size

        #povecavamo vrednost u odg binu
        histograms[0][r_bin] += 1 #red
        histograms[1][g_bin] += 1 #green
        histograms[2][b_bin] += 1 #blue
        return histograms


    pixels = map(lambda y: map(lambda x: image.getpixel((x, y)), custom_range(width)), custom_range(height))
    flat_pixels = reduce(lambda acc, row: acc + list(row), pixels, [])
    #dobijamo jednodimenzionalni niz piksela

    r_hist, g_hist, b_hist = reduce(update_histogram, flat_pixels, initial_histograms)

    total_pixels = width * height
    # Normalizacija histograma
    r_hist = list(map(lambda x: x / total_pixels, r_hist))
    g_hist = list(map(lambda x: x / total_pixels, g_hist))
    b_hist = list(map(lambda x: x / total_pixels, b_hist))

    return [r_hist, g_hist, b_hist]


#dodajemo nizove - koristimo kako bismo sabrali histograme
def add_arrays(a1, a2):
    sum = list(map(lambda x, y: (float(x)+ float(y)), a1, a2))
    return sum

#sabiramo histograme tako sto koristimo odgovarajuce nizove koje sabiramo
def add_hist(hist1,hist2):
    suma = list(map(lambda x, y: add_arrays(x,y), hist1,hist2))
    return suma

#2.
def average_histogram(lista):
    #classy = set(map(lambda x: x[0], lista))# promeniti na map
    classy = reduce(lambda acc, x: (acc.add(x[0]) or acc), lista, set())

    def find_avg(cls):
        #filtered_images = list(filter(lambda x: x[0] == cls, lista))
        filtered_images = list(reduce(lambda acc, x: acc if x[0] != cls else acc + [x], lista, []))

        histograms = list(map(lambda x: calculate_normalized_bins_histograms(x[1]), filtered_images))
        histograms_length = list_length(histograms)

        if list_length(histograms) != 1:
            sum_hist = reduce(lambda acc, hist: add_hist(acc, hist), histograms)#RADIIII!!!
        else:
            sum_hist = histograms[0]
        avg_histogram = list(map(lambda row: list(map(lambda x: x / histograms_length, row)), sum_hist))
        return (cls, avg_histogram)#vraca par, avg_hist
    return list(map(find_avg, classy))#poziv funkcije find_avg

#3.
def cosine_similarity(hist1, hist2):#RADIII

    hist1 = np.array(hist1) if not isinstance(hist1, np.ndarray) else hist1
    hist2 = np.array(hist2) if not isinstance(hist2, np.ndarray) else hist2
    flat_hist1 = hist1.flatten()
    flat_hist2 = hist2.flatten()

    #nalazimo proizvod svaka 2 el u nizovima
    proizvod = list(map(lambda a, b: float(a * b), flat_hist1, flat_hist2))
    #sabiramo da bismo dobili skalarni proizvod
    dot_product = reduce(lambda x, y: x + y, proizvod)

    norm1 = reduce(lambda x,y: x + y*y, flat_hist1, 0)
    hist1_norm = math.sqrt(norm1)

    norm2 = reduce(lambda x, y: x + y*y, flat_hist2, 0)
    hist2_norm = math.sqrt(norm2)

    if hist1_norm == 0 or hist2_norm == 0:
        return 0.0

    similarity = dot_product / (hist1_norm * hist2_norm)

    return similarity

def custom_max(similarity):
    return reduce(lambda x, y: x if x[1] > y[1] else y, similarity)

def compare_similarity(current_hist, avg_hist):

    print(current_hist)
    print("----------------------")
    print(avg_hist)
    #posto nam je avg_hist oblika [('airplane', 0.8174248557792346), ('automobile', 0.8572460315450375)...
    #x[0] nam je "kljuc", a x [1] nam je broj koji poredimo
    similarities = list(
        map(lambda x: (x[0], (cosine_similarity(current_hist,x[1]))), avg_hist)
    )
    print(similarities)

    best_match = custom_max(similarities)
    return best_match
#4
def classify_image(image_path, images):
    try:
        histogram = calculate_normalized_bins_histograms(image_path)
    except FileNotFoundError:
        print("Fajl ne postoji")
        return
    #print(histogram)

    best_match = compare_similarity(histogram, average_histogram(images))

    return best_match


if __name__ == "__main__":
    #ucitavanje svih slika iz "baze"
    current_dir = os.getcwd()
    data_set_path = os.path.join(current_dir, "organized_cifar10")
    images = load_images(data_set_path)

    test_path = os.path.join(current_dir, "test")
    while(True):
        slika = input("> ")
        if(slika != "exit"):
            slika_path = os.path.join(test_path, slika)
            print(slika_path)
            best = classify_image(slika_path, images)

            print(best)
        else:
            break