import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# computing quantisized energy
def q_energy(energy, dN_dE, q_rate = 100):
    start = 0
    end   = 16000

    bins = np.arange(start, end + q_rate, q_rate)

    digitized = np.digitize(energy, bins)
    centers = bins[:-1] + q_rate / 2

    rates = []

    for i in range(1, len(bins)):
        mask = (digitized == i)
        if mask.any():
            rates.append(np.mean(dN_dE[mask]))
        else:
            rates.append(0.0)

    return centers, rates

# plot and save spectrum of isotope
def plot_data(isotope, quantized_data, save_fig = False):
    
    plt.figure(figsize=(15, 6))
    plt.errorbar(quantized_data['Energy'], quantized_data['dN/dE'], ecolor='r')
    plt.xlabel('Energy (keV)')
    plt.ylabel('dN/dE')
    plt.title(f'Energy Spectrum of {isotope}')
    plt.grid(True)

    if save_fig:
        filename = f'{isotope}.png'
        plot_path = os.path.join('savedplots/', filename)
        plt.savefig(plot_path)

    plt.show()


# get q_energy dataset from file
def get_q_data_from_file(file_path, NUM_OF_ISOTOPES = 1, energy_shift = 0, q_rate = 100, norm = 1):
    data = pd.read_csv(file_path, sep='\s+', header=None, names=['Energy', 'dN/dE', 'Uncertainty'])
    quantized_energy, quantized_dN_dE = q_energy(energy_shift - data['Energy'], data['dN/dE'] * NUM_OF_ISOTOPES)
    
    quantized_data = pd.DataFrame({
        'Energy': quantized_energy,
        'dN/dE': quantized_dN_dE
    })

    quantized_data['dN/dE'] /= norm
    return quantized_data

# get q_energy dataset
def get_q_data(data, NUM_OF_ISOTOPES = 1, energy_shift = 0, q_rate = 100, norm = 1):
    quantized_energy, quantized_dN_dE = q_energy(energy_shift - data['Energy'], data['dN/dE'] * NUM_OF_ISOTOPES)
    
    quantized_data = pd.DataFrame({
        'Energy': quantized_energy,
        'dN/dE': quantized_dN_dE
    })

    quantized_data['dN/dE'] /= norm
    return quantized_data

# compute and plot total spec based on all isotopes
def find_total_spectrum(result_data, save_fig = False):
    result_data
    spectra = []

    for iso, df in result_data.items():
        s = df.set_index('Energy')['dN/dE'].rename(iso)
        spectra.append(s)

    all_spectra = pd.concat(spectra, axis=1, sort=True).fillna(0)

    all_spectra['total_dN/dE'] = all_spectra.sum(axis=1)
    all_spectra['total_dN/dE']
    total_spectrum = all_spectra[['total_dN/dE']].reset_index()

    plt.figure(figsize=(15, 6))
    plt.title('Energy Spectrum')
    plt.errorbar(total_spectrum['Energy'], total_spectrum['total_dN/dE'], ecolor='r', label = 'sum')
    for isotope in result_data:
        data = result_data[isotope] 
        plt.errorbar(data['Energy'], data['dN/dE'], ecolor='r', label = isotope)
        
    plt.xlabel('Energy (keV)')
    plt.ylabel('dN/dE')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig("savedplots/sum.png")

    plt.show()

    return total_spectrum

# find integral of total spec
def integrate(left, right, x, y, delta_rate, norm):
    result = 0
    for i in range(len(x)):
        energy, spec = x[i], y[i]
        if left <= energy <= right:
            result += spec * delta_rate
    
    return result / norm