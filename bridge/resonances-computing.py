import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, fftfreq
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
import time
import psutil  # Para verificar la memoria disponible
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py



def process_file(file):
    try:
        with h5py.File(file, 'r') as f:
            data = np.array(f['/Acquisition/Raw[0]/RawData'])
            ft = fftshift(fft(data[:, 1100]))
            abs_val = np.abs(ft)
            return abs_val
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None

def check_memory():
    # Verifica la memoria disponible en el sistema
    memory = psutil.virtual_memory()
    return memory.available > 1e9  # 1 GB de espacio libre mínimo requerido

def main():
    file_list1 = glob('/auto/petasaur-wd15/rainier-10-14-2023-drive1/decimator_*')
    
    file_list = file_list1
    file_list.sort()

    #L_total = len(file_list)
    file_list = file_list[100:-1]
    L_total = len(file_list)
    print(f"Total files to process: {L_total}")

    # Leer un archivo de muestra para conocer las dimensiones
    sample_file = file_list[0]
    with h5py.File(sample_file, 'r') as d:
        sample_data = np.array(d['/Acquisition/Raw[0]/RawData'])
        num_rows = sample_data.shape[0]

    # Parámetros de batch
    batch_size = 200  # Número de archivos a procesar por batch
    num_batches = (L_total + batch_size - 1) // batch_size  # Número de batches

    mat = np.zeros((num_rows, 0))  # Matriz para almacenar todos los resultados

    start_time = time.time()

    for batch_index in range(num_batches):
        if not check_memory():
            print("Memoria insuficiente. Terminando proceso.")
            break

        start_idx = batch_index * batch_size
        end_idx = min((batch_index + 1) * batch_size, L_total)
        batch_files = file_list[start_idx:end_idx]

        print(f"Processing batch {batch_index + 1}/{num_batches} with files {start_idx} to {end_idx}")

        with Pool() as pool:
            results = list(tqdm(pool.imap(process_file, batch_files), total=len(batch_files)))

        # Filtra resultados válidos
        batch_mat = np.array([res for res in results if res is not None and res.shape[0] == num_rows]).T

        if batch_mat.size == 0:
            print(f"No valid data in batch {batch_index + 1}. Skipping batch.")
            continue

        mat = np.hstack((mat, batch_mat))  # Añade los resultados a la matriz principal

        # Guardar resultados intermedios
        # Cambiar el path para guardar resultados intermedios
        output_path = '/data/data4/veronica-scratch-rainier/bridge-resonance/intermediate_results_batch_'  # Cambia esta ruta a     donde desees guardar
        np.save(f'{output_path}{batch_index + 1}.npy', mat)

        

    # Generar el eje de frecuencia
    f = fftshift(fftfreq(num_rows, d=1/200))

    # Generar el eje de tiempo en días
    y_axis = np.arange(0, mat.shape[1]) / 1440

    if mat.size == 0:
        print("No data to plot. Exiting.")
        return

    # Graficar el resultado
    plt.subplots(figsize=(16, 9))
    plt.pcolormesh(f, y_axis, mat.T, vmin=0, vmax=1e3, cmap='gray_r', shading='nearest')
    plt.xlim([0, 20])
    plt.xlabel('Frequency (Hz)', fontsize=25)
    plt.ylabel('Days', fontsize=25)
    cbar = plt.colorbar()
    cbar.set_label('Amplitude', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plot_path = 'frequency_vs_days_plot_2folder.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    end_time = time.time()
    print(f"Plot saved to {plot_path}")
    print(f"Execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()
