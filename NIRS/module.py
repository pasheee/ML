import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def q_energy(energy, dN_dE, q_rate = 100):
    """
    Quantizes energy data into specified bins and computes the average rate for each bin.

    Parameters:
    energy (array-like): Array of energy values.
    dN_dE (array-like): Array of differential number density values corresponding to the energies.
    q_rate (int, optional): The width of each energy bin. Default is 100.

    Returns:
    tuple: A tuple containing:
        - centers (array): The center values of the energy bins.
        - rates (list): The average rates in each bin.
    """

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


def plot_data(isotope, quantized_data, save_fig = False):
    """
    Plots the energy spectrum of a given isotope using quantized data.

    Parameters:
    isotope (str): Name of the isotope to be plotted.
    quantized_data (pd.DataFrame): DataFrame containing quantized energy values and corresponding dN/dE rates.
    save_fig (bool, optional): If True, saves the plot as a PNG image in the 'savedplots/' directory. Default is False.
    """

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


def get_q_data_from_file(file_path, NUM_OF_ISOTOPES = 1, energy_shift = 0, q_rate = 100, norm = 1):
    """
    Reads energy spectrum data from a file, applies quantization, and returns the quantized data.

    Parameters:
    file_path (str): Path to the file containing the data. The file should have columns for 'Energy', 'dN/dE', and 'Uncertainty'.
    NUM_OF_ISOTOPES (int, optional): Number of isotopes to consider for the dN/dE calculation. Default is 1.
    energy_shift (float, optional): Energy value to shift the spectrum by before quantizing. Default is 0.
    q_rate (int, optional): Quantization rate for processing energy data (not directly used in this function). Default is 100.
    norm (float, optional): Normalization factor for the quantized dN/dE values. Default is 1.

    Returns:
    pd.DataFrame: A DataFrame containing quantized 'Energy' and 'dN/dE' columns.
    """

    data = pd.read_csv(file_path, sep='\s+', header=None, names=['Energy', 'dN/dE', 'Uncertainty'])
    quantized_energy, quantized_dN_dE = q_energy(energy_shift - data['Energy'], data['dN/dE'] * NUM_OF_ISOTOPES)
    
    quantized_data = pd.DataFrame({
        'Energy': quantized_energy,
        'dN/dE': quantized_dN_dE
    })

    quantized_data['dN/dE'] /= norm
    return quantized_data


def get_q_data(data, NUM_OF_ISOTOPES = 1, energy_shift = 0, q_rate = 100, norm = 1):
    """
    Quantizes the energy spectrum data given in a DataFrame and returns the quantized data.

    Parameters:
    data (pd.DataFrame): DataFrame containing 'Energy' and 'dN/dE' columns.
    NUM_OF_ISOTOPES (int, optional): Number of isotopes to consider for the dN/dE calculation. Default is 1.
    energy_shift (float, optional): Energy value to shift the spectrum by before quantizing. Default is 0.
    q_rate (int, optional): Quantization rate for processing energy data (not directly used in this function). Default is 100.
    norm (float, optional): Normalization factor for the quantized dN/dE values. Default is 1.

    Returns:
    pd.DataFrame: A DataFrame containing quantized 'Energy' and 'dN/dE' columns.
    """
    quantized_energy, quantized_dN_dE = q_energy(energy_shift - data['Energy'], data['dN/dE'] * NUM_OF_ISOTOPES)
    
    quantized_data = pd.DataFrame({
        'Energy': quantized_energy,
        'dN/dE': quantized_dN_dE
    })

    quantized_data['dN/dE'] /= norm
    return quantized_data


def find_total_spectrum(result_data, plot = False, save_fig = False):
    """
    Computes and plots the total energy spectrum based on all isotopes in the given result_data dictionary.

    Parameters:
    result_data (dict): A dictionary containing the energy spectrum data for each isotope, where each value is a DataFrame with 'Energy' and 'dN/dE' columns.
    save_fig (bool, optional): Whether to save the plot as an image file. Default is False.

    Returns:
    pd.DataFrame: A DataFrame containing the total energy spectrum with 'Energy' and 'total_dN/dE' columns.
    """

    result_data
    spectra = []

    for iso, df in result_data.items():
        s = df.set_index('Energy')['dN/dE'].rename(iso)
        spectra.append(s)

    all_spectra = pd.concat(spectra, axis=1, sort=True).fillna(0)

    all_spectra['total_dN/dE'] = all_spectra.sum(axis=1)
    all_spectra['total_dN/dE']
    total_spectrum = all_spectra[['total_dN/dE']].reset_index()

    if plot:
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
        if save_fig:
            plt.savefig("savedplots/sum.png")

        plt.show()

    return total_spectrum


def integrate(left, right, x, y, delta_rate, norm = 1):
    
    """
    Computes the integral of the total energy spectrum from left to right.

    Parameters:
    left (float): Lower energy bound for the integral.
    right (float): Upper energy bound for the integral.
    x (array-like): Array of energy values.
    y (array-like): Array of corresponding dN/dE rates.
    delta_rate (float): Energy bin width.
    norm (float): Normalization factor.

    Returns:
    float: The integral of the total energy spectrum from left to right.
    """

    result = 0
    for i in range(len(x)):
        energy, spec = x[i], y[i]
        if left <= energy <= right:
            result += spec * delta_rate
    
    return result / norm


