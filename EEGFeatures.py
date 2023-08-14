#
# Uses EEGExtract: https://github.com/physionetchallenges/python-example-2023
#
import EEGExtract as eeg
import numpy as np

def extract_all_eeg_features(data, sampling_frequency):
    assert(data.ndim == 3)
    feature_list = []
    fs = sampling_frequency

    ### COMPLEXITY FEATURES ###

    # Shannon Entropy
    shannon_res = eeg.shannonEntropy(data, 200, 200, 2)
    feature_list.append(np.median(shannon_res, axis=1))


    # Subband Information Quantity
    # delta (0.5-4 Hz)
    eegData_delta = eeg.filt_data(data, 0.5, 4, fs)
    shannon_res_delta = eeg.shannonEntropy(eegData_delta, -200, 200, 2)
    feature_list.append(np.median(shannon_res_delta, axis=1))

    # theta (4-8 Hz)
    eegData_theta = eeg.filt_data(data, 4, 8, fs)
    shannon_res_theta = eeg.shannonEntropy(eegData_theta, -200, 200, 2)
    feature_list.append(np.median(shannon_res_theta, axis=1))

    # gamma (30-100 Hz)
    eegData_gamma = eeg.filt_data(data, 30, 100, fs)
    shannon_res_gamma = eeg.shannonEntropy(eegData_gamma, -200, 200, 2)
    feature_list.append(np.median(shannon_res_gamma, axis=1))

    # δ band Power
    bandPwr_delta = eeg.bandPower(data, 0.5, 4, fs)
    feature_list.append(np.median(bandPwr_delta, axis=1))

    # θ band Power
    bandPwr_theta = eeg.bandPower(data, 4, 8, fs)
    feature_list.append(np.median(bandPwr_theta, axis=1))

    # α band Power
    bandPwr_alpha = eeg.bandPower(data, 8, 12, fs)
    feature_list.append(np.median(bandPwr_alpha, axis=1))
    
    # β band Power
    bandPwr_beta = eeg.bandPower(data, 12, 30, fs)
    feature_list.append(np.median(bandPwr_beta, axis=1))

    # γ band Power
    bandPwr_gamma = eeg.bandPower(data, 30, 100, fs)
    feature_list.append(np.median(bandPwr_gamma, axis=1))


    # Standard Deviation
    std_res = eeg.eegStd(data)
    feature_list.append(np.median(std_res, axis=1))

    # Regularity (burst-suppression)
    regularity_res = eeg.eegRegularity(data, int(fs))
    feature_list.append(np.median(regularity_res, axis=1))

    # Voltage < 5μ
    volt05_res = eeg.eegVoltage(data, voltage=5)
    feature_list.append(np.median(volt05_res, axis=1))
    
    # Voltage < 10μ
    volt10_res = eeg.eegVoltage(data, voltage=10)
    feature_list.append(np.median(volt10_res, axis=1))

    # Voltage < 20μ
    volt20_res = eeg.eegVoltage(data, voltage=20)
    feature_list.append(np.median(volt20_res, axis=1))

    # Diffuse Slowing
    df_res = eeg.diffuseSlowing(data)
    feature_list.append(np.median(df_res, axis=1))

    # Spikes
    minNumSamples = int(70*fs/1000)
    spikeNum_res = eeg.spikeNum(data, minNumSamples)
    feature_list.append(np.median(spikeNum_res, axis=1))

    # Delta burst after Spike
    deltaBurst_res = eeg.burstAfterSpike(data, eegData_delta, minNumSamples=7, stdAway = 3)
    feature_list.append(np.median(deltaBurst_res, axis=1))

    # Sharp spike
    sharpSpike_res = eeg.shortSpikeNum(data, minNumSamples)
    feature_list.append(np.median(sharpSpike_res, axis=1))

    # Number of Bursts
    numBursts_res = eeg.numBursts(data, fs)
    feature_list.append(np.median(numBursts_res, axis=1))

    # Number of Suppressions
    numSupps_res = eeg.numSuppressions(data, fs)
    feature_list.append(np.median(numSupps_res, axis=1))

    return np.array(feature_list).T