import multiprocessing
import threading
import json
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import queue
import time
import os
from multiprocessing import Pool, Value


class Slika:
    #Pslika je prosla slika
    def __init__(self, pSlika, iArray, putanja, task_id, time):
        self.iArray = iArray
        self.putanja = putanja
        self.task_id = task_id
        #skoro sve je slicno samo sam dodao ovu istoriju.
        #Trebalo bi da je ovo ok za cuvanje
        if pSlika is None:
            self.history = []
        else:
            tmp = pSlika.history.copy()
            tmp.append([pSlika.putanja, pSlika.task_id])
            self.history = tmp
        #self.history = Rslika[pSlika].history.append([pSlika, Rslika[pSlika].task_id])
        self.tasks = []
        self.marked_for_deletion = False
        self.applied_filters = []
        self.processing_time = time
        self.size_before = None
        self.size_after = None

class Zadatak:
    def __init__(self, idSlike, filter, sigma):
        self.idSlike = idSlike
        self.filter = filter
        self.sigma = sigma
        self.status = "cekanje"
        self.condition = threading.Condition()



#Ovo je dict za registar i brojac za idove
Rslika = {}
Rzadatak = {}
s_brojac = 0
z_brojac = 0
threads = []

zadaci_katanac = threading.Lock()
slike_katanac = threading.Lock()
condidition_dict = {}

izvrseni_queue = queue.Queue()
command_queue = queue.Queue()
output_queue = queue.Queue()
running = True

def grayscale(image_array):
    red_channel = image_array[..., 0]
    green_channel = image_array[..., 1]
    blue_channel = image_array[..., 2]

    grayscale_image = (red_channel * 0.299 + green_channel * 0.587 + blue_channel * 0.114)
    return grayscale_image.astype(np.uint8)

def gaussian_blur(image_array, sigma=1):

    red_channel = gaussian_filter(image_array[..., 0], sigma=sigma)
    green_channel = gaussian_filter(image_array[..., 1], sigma=sigma)
    blue_channel = gaussian_filter(image_array[..., 2], sigma=sigma)

    blurred_image = np.zeros_like(image_array)
    blurred_image[..., 0] = red_channel
    blurred_image[..., 1] = green_channel
    blurred_image[..., 2] = blue_channel

    if image_array.shape[-1] == 4:
        alpha_channel = image_array[..., 3]
        blurred_image[..., 3] = alpha_channel

    blurred_image = np.clip(blurred_image, 0, 255)

    return blurred_image.astype(np.uint8)

def adjust_brightness(image_array, factor=1.0):
    mean_intensity = np.mean(image_array, axis=(0, 1), keepdims=True)  # Računanje srednje vrednosti piksela
    image_array = (image_array - mean_intensity) * factor + mean_intensity  # Skaliranje prema srednjoj vrednosti

    adjusted_image = np.clip(image_array, 0, 255)

    return adjusted_image.astype(np.uint8)


def save_image(new_image, new_image_path):
    new_image_pil = Image.fromarray(new_image)
    new_image_pil.save(new_image_path)

def update_photo_registry(pSlika, image_path, task_id, timex):
    global Rslika, s_brojac, slike_katanac, Rzadatak
    img = load_image(image_path)
    nSlika = Slika(pSlika, img, image_path, task_id, timex)

    nSlika.size_after = os.path.getsize(image_path) / 1024
    if pSlika is not None:
        nSlika.size_before = pSlika.size_after
        tmp = pSlika.applied_filters.copy()
        tmp.append(Rzadatak[task_id].filter)
        nSlika.applied_filters = tmp
    else:
        nSlika.size_before = nSlika.size_after

    Rslika[s_brojac] = nSlika

    for zad in Rzadatak.values():

        if zad.idSlike == s_brojac:
            with zad.condition:
                print(zad.idSlike)
                zad.condition.notify_all()

    s_brojac += 1
    print(f"Nova slika dodata u registar slika sa iD {s_brojac-1}")
    print(Rslika)
    return nSlika

def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)


def add(image_path):        ## add command
    global Rslika
    try:
        nSlika = update_photo_registry(None, image_path, None, 0)
        print("adding image... " + nSlika.putanja)
        return nSlika
    except FileNotFoundError:
        print("Slika ne postoji :(")


def delete(idSlike):
    global Rslika, Rzadatak, slike_katanac, zadaci_katanac, condidition_dict

    if int(idSlike) not in Rslika:
        print("Slika ne postoji")
        return

    Rslika[int(idSlike)].marked_for_deletion = True

    print("Brisanje slike...")
    for zad in Rzadatak.values():
        if zad.idSlike == int(idSlike):
            with zad.condition:
                while zad.status != "zavrseno":
                    print(f"Čekam da se zadatak sa slikom {idSlike} završi...")
                    zad.condition.wait()

    with slike_katanac:
        real_path = os.path.join(os.getcwd(),Rslika[int(idSlike)].putanja)
        os.remove(real_path)
        Rslika.pop(int(idSlike))
    print(Rslika)

def describe(idSlika):
    global Rslika
    id_slike = int(idSlika)
    time.sleep(0.2)
    output_queue.put(Rslika[id_slike].history)

    return

def list():
    global Rslika, Rzadatak, zadaci_katanac
    for key, value in Rslika.items():
        result = (key, value.putanja)
        output_queue.put(result)


def zadatak_callback(result):
    global zadaci_katanac
    if result is not None:
        izvrseni_queue.put(result)

    return None


def transform_image(task_id, image_id, iArray, filter, sigma, new_image_path):
    global Rslika, Rzadatak
    tStart = time.time()
    if filter == "grayscale":
        nova_slika = grayscale(iArray)
    elif filter == "gaussian_blur":
        nova_slika = gaussian_blur(iArray, sigma)
    elif filter == "adjust_brightness":
        nova_slika = adjust_brightness(iArray,sigma)
    else:
        print("Unknown filter")

    print("cuva sliku...")
    time.sleep(2)
    return (task_id, nova_slika, new_image_path, image_id, tStart)

def process(json_path):     ## process command
    global Rslika, Rzadatak, z_brojac, s_brojac

    print("obrada slike...")
    with open(json_path) as f:
        try:
            params = json.load(f)
        except FileNotFoundError:
            print("Ne postoji")

        image_id = int(params.get("slika_id"))
        filter = params.get("filter")

        if image_id is None or filter is None:
            print("Los format")
            return

        if image_id in Rslika.keys():
            if Rslika[image_id].marked_for_deletion is True:
                print("Ne moze")
                return

        try:
            sigma = params.get("sigma")
        except KeyError:
            sigma = 0


    with zadaci_katanac:
        zadatak = Zadatak(image_id, filter, sigma)
        Rzadatak[z_brojac] = zadatak

    with zadatak.condition:
        if image_id not in Rslika.keys():
            print(f"Slika {image_id} nije još kreirana, čekam...")
            zadatak.condition.wait()


    tr_direktorijum = os.getcwd()
    folder = os.path.join(tr_direktorijum, "Slike")
    os.makedirs(folder, exist_ok=True)
    new_path = f"{folder}/Slika_{image_id}_{z_brojac}.jpg"

    with slike_katanac:

        Rslika[image_id].tasks.append(z_brojac)
        iArray = Rslika[image_id].iArray
    time.sleep(1.5)
    pool.apply_async(transform_image, args=(z_brojac, image_id, iArray, filter, sigma, new_path), callback=zadatak_callback)
    z_brojac += 1


def exit_program():
    global running
    running = False
    print("izlazak iz programa...")

    return

def izvrseni_zadaci():
    global izvrseni_queue, Rslika, s_brojac,Rzadatak, z_brojac, zadaci_katanac
    while running:
        try:

            task_id, nova_slika, new_image_path, image_id, tStart = izvrseni_queue.get(timeout=1)
            save_image(nova_slika, new_image_path)
            update_photo_registry(Rslika[image_id], new_image_path, task_id, time.time() - tStart)
            with zadaci_katanac:
                Rzadatak[task_id].status = "zavrseno"
                with Rzadatak[task_id].condition:
                    Rzadatak[task_id].condition.notify_all()
            print(f"Zadatak {task_id} je završen i sačuvan na {new_image_path}")
            izvrseni_queue.task_done()
        except queue.Empty:
            continue
#obradjujemo komande iz reda
def command_handler():
    while running or not command_queue.empty():
        try:
            command = command_queue.get(timeout=1)
            parts = command.split()
            action = parts[0]


            if action == "list":
                p = threading.Thread(target=list)
                p.start()
            elif action == "describe" and len(parts) > 1:
                p = threading.Thread(target=describe, args=(parts[1]))
                p.start()
            else:
                output_queue.put("Nepoznata komanda\n")

            p.join()
        except queue.Empty:
            continue

#citamo komande iz reda i saljemo na output
def output_thread():
    while running or not output_queue.empty():
        try:
            message = output_queue.get(timeout=1)
            print(message)
        except queue.Empty:
            continue

def main(command):
    global image_counter, task_counter, running, threads
    parts = command.split()
    action = parts[0]

    if action == "add" and len(parts) > 1:
        print(f"dodajem sliku {parts[1]}\n")
        t = threading.Thread(target=add, args=(parts[1],))
        t.start()
        threads.append(t)
    elif action == "process" and len(parts) > 1:
        t = threading.Thread(target=process, args=(parts[1],))
        t.start()
        threads.append(t)
    elif action == "delete" and len(parts) > 1:
        t = threading.Thread(target=delete, args=(parts[1],))
        t.start()
        threads.append(t)
    elif action == "list":
        command_queue.put("list")
    elif action == "describe" and len(parts) > 1:
        command_queue.put(f"describe {parts[1]}")  # Komanda "describe" ide u red
    elif action == "exit":
        exit_program()
    else:
        print("Unknown command\n")



if __name__ == "__main__":

    #pokrecu se niti za obradu komandi i red za ispis poruka
    command_handler_thread = threading.Thread(target=command_handler)
    output_thread_instance = threading.Thread(target=output_thread)
    zavrseni_zadaci_thread = threading.Thread(target=izvrseni_zadaci)
    command_handler_thread.start()
    output_thread_instance.start()
    zavrseni_zadaci_thread.start()
    pool = multiprocessing.Pool(2)
    with Pool(4) as pool:
        while running:
            command = input("> ")
            main(command)
            for t in threads:
                if len(threads) > 1:
                    t.join()
            if not running:
                break

    pool.close()
    pool.join()
    command_handler_thread.join()
    output_thread_instance.join()
    zavrseni_zadaci_thread.join()

