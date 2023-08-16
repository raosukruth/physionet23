#
# Uses EEGExtract: https://github.com/sari-saba-sadiya/EEGExtract/tree/main
#
import EEGExtract as eeg
import numpy as np
import time
from multiprocessing import Pool

# Shannon Entropy
def extract_shannon_entropy(data, sampling_frequency):
    return np.median(eeg.shannonEntropy(data, 200, 200, 2), axis=1)

# Subband Information Quantity
# delta (0.5-4 Hz)
def extract_delta(data, sampling_frequency):
    eegData_delta = eeg.filt_data(data, 0.5, 4, sampling_frequency)
    return np.median(eeg.shannonEntropy(eegData_delta, -200, 200, 2), axis=1)

# theta (4-8 Hz)
def extract_theta(data, sampling_frequency):
    eegData_theta = eeg.filt_data(data, 4, 8, sampling_frequency)
    return np.median(eeg.shannonEntropy(eegData_theta, -200, 200, 2), axis=1)

# gamma (30-100 Hz)
def extract_gamma(data, sampling_frequency):
    eegData_gamma = eeg.filt_data(data, 30, 100, sampling_frequency)
    return np.median(eeg.shannonEntropy(eegData_gamma, -200, 200, 2), axis=1)

# δ band Power
def extract_band_delta(data, sampling_frequency):
    bandPwr_delta = eeg.bandPower(data, 0.5, 4, sampling_frequency)
    return np.median(bandPwr_delta, axis=1)

# θ band Power
def extract_band_theta(data, sampling_frequency):
    bandPwr_theta = eeg.bandPower(data, 4, 8, sampling_frequency)
    return np.median(bandPwr_theta, axis=1)

# α band Power
def extract_band_alpha(data, sampling_frequency):
    bandPwr_alpha = eeg.bandPower(data, 8, 12, sampling_frequency)
    return np.median(bandPwr_alpha, axis=1)

# β band Power
def extract_band_beta(data, sampling_frequency):
    bandPwr_beta = eeg.bandPower(data, 12, 30, sampling_frequency)
    return np.median(bandPwr_beta, axis=1)

# γ band Power
def extract_band_gamma(data, sampling_frequency):
    bandPwr_gamma = eeg.bandPower(data, 30, 100, sampling_frequency)
    return np.median(bandPwr_gamma, axis=1)

# Standard Deviation
def extract_standard_dev(data, sampling_frequency):
    std_res = eeg.eegStd(data)
    return np.median(std_res, axis=1)

# Regularity (burst-suppression)
def extract_regularity(data, sampling_frequency):
    regularity_res = eeg.eegRegularity(data, int(sampling_frequency))
    return np.median(regularity_res, axis=1)

# Voltage < 5μ
def extract_volt05(data, sampling_frequency):
    volt05_res = eeg.eegVoltage(data, voltage=5)
    return np.median(volt05_res, axis=1)

# Voltage < 10μ
def extract_volt10(data, sampling_frequency):
    volt10_res = eeg.eegVoltage(data, voltage=10)
    return np.median(volt10_res, axis=1)

# Voltage < 20μ
def extract_volt20(data, sampling_frequency):
    volt20_res = eeg.eegVoltage(data, voltage=20)
    return np.median(volt20_res, axis=1)

# Diffuse Slowing
def extract_diffuse_slowing(data, sampling_frequency):
    df_res = eeg.diffuseSlowing(data)
    return np.median(df_res, axis=1)

# Spikes
def extract_spikes(data, sampling_frequency):
    minNumSamples = int(70 * sampling_frequency / 1000)
    spikeNum_res = eeg.spikeNum(data, minNumSamples)
    return np.median(spikeNum_res, axis=1)

# Delta burst after Spike
def extract_delta_after_spike(data, sampling_frequency):
    eegData_delta = eeg.filt_data(data, 0.5, 4, sampling_frequency)
    deltaBurst_res = eeg.burstAfterSpike(data, eegData_delta, minNumSamples=7, stdAway=3)
    return np.median(deltaBurst_res, axis=1)

# Sharp spike
def extract_sharp_spike(data, sampling_frequency):
    minNumSamples = int(70 * sampling_frequency / 1000)
    sharpSpike_res = eeg.shortSpikeNum(data, minNumSamples)
    return np.median(sharpSpike_res, axis=1)

# Number of Bursts
def extract_number_of_bursts(data, sampling_frequency):
    numBursts_res = eeg.numBursts(data, sampling_frequency)
    return np.median(numBursts_res, axis=1)

# Number of Suppressions
def extract_number_of_suppressions(data, sampling_frequency):
    numSupps_res = eeg.numSuppressions(data, sampling_frequency)
    return np.median(numSupps_res, axis=1)

def extract_feature(extract_func, data, sampling_frequency):
    return extract_func(data, sampling_frequency)

def extract_all_eeg_features(data, sampling_frequency):
    feature_args = [
        (extract_shannon_entropy, data, sampling_frequency),
        (extract_delta, data, sampling_frequency),
        (extract_theta, data, sampling_frequency),
        (extract_gamma, data, sampling_frequency),
        (extract_band_delta, data, sampling_frequency),
        (extract_band_theta, data, sampling_frequency),
        (extract_band_alpha, data, sampling_frequency),
        (extract_band_beta, data, sampling_frequency),
        (extract_band_gamma, data, sampling_frequency),
        (extract_standard_dev, data, sampling_frequency),
        (extract_regularity, data, sampling_frequency),
        (extract_volt05, data, sampling_frequency),
        (extract_volt10, data, sampling_frequency),
        (extract_volt20, data, sampling_frequency),
        (extract_diffuse_slowing, data, sampling_frequency),
        (extract_spikes, data, sampling_frequency),
        (extract_delta_after_spike, data, sampling_frequency),
        (extract_sharp_spike, data, sampling_frequency),
        (extract_number_of_bursts, data, sampling_frequency),
        (extract_number_of_suppressions, data, sampling_frequency)
    ]
    result = None
    with Pool(processes=20) as p:
        result = p.starmap(extract_feature, feature_args)
    return np.array(result).T