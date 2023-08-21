#!/usr/bin/env python

#
# Based on team code: https://github.com/physionetchallenges/python-example-2023
#

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import EEGFeatures as eegf
import tensorflow as tf
import time
import os
import numpy as np
from multiprocessing import Pool, process

def process_patient(i, data_folder, verbose, patient_ids):
    proc = process.current_process()
    proc._config['daemon'] = False
    if verbose >= 2:
        print('    {}/{}...'.format(i+1, num_patients))
    current_features = get_features(data_folder, patient_ids[i])
    
    # Extract labels.
    patient_metadata = load_challenge_data(data_folder, patient_ids[i])
    current_outcome = get_outcome(patient_metadata)
    current_cpc = get_cpc(patient_metadata)
    return current_features, current_outcome, current_cpc

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

def train_challenge_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose >= 1:
        print('Finding the Challenge data...')

    patient_ids = find_data_folders(data_folder)
    
    num_patients = len(patient_ids)

    if num_patients == 0:
        raise FileNotFoundError('No data was provided.')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Extract the features and labels.
    if verbose >= 1:
        print('Extracting features and labels from the Challenge data...')
    
    with Pool(processes=16) as p:
        args = [(i, data_folder, verbose, patient_ids) for i in range(num_patients)]
        result = p.starmap(process_patient, args)

    features, outcomes, cpcs = zip(*result)
    
    # Train the models.
    if verbose >= 1:
        print('Training the Challenge model on the Challenge data...')

    # Load features, outcomes, and cpcs data
    features = np.vstack(features)
    outcomes = np.vstack(outcomes)
    cpcs = np.vstack(cpcs)

    # Impute missing features
    imputer = SimpleImputer().fit(features)
    features = imputer.transform(features)
    assert(np.isnan(features).any() == False)

    # Define neural network architecture using TensorFlow
    input_shape = features.shape[1]

    outcome_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.softmax)
    ])

    cpcs_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.softmax)
    ])

    outcome_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    cpcs_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    outcome_model.fit(features, outcomes, epochs=1000, batch_size=32, verbose='auto') 
    cpcs_model.fit(features, cpcs, epochs=1000, batch_size=32, verbose='auto') 

    # Save the models.
    print(time.asctime(), "Begin saving the model")
    save_challenge_model(model_folder, imputer, outcome_model, cpcs_model)

    if verbose >= 1:
        print('Done.')


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']
    
    # Extract features.
    features = get_features(data_folder, patient_id)
    if features is None:
        return None, 0, 1
    features = features.reshape(1, -1)

    # Impute missing data.
    features = imputer.transform(features)
    assert(np.isnan(features).any() == False)


    # Apply models to features.
    # outcome = outcome_model.predict(features)[0]
    # outcome_probability = tf.nn.softmax(outcome, axis=0)[0].numpy()
    # cpc = cpc_model.predict(features)[0]

    outcome = outcome_model.predict(features)[0]
    #outcome_probability = outcome_model.predict_proba(features)[0, 0]
    outcome_probabilities = outcome_model.predict(features)
    outcome_probability = outcome_probabilities[0]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.5, 100.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    if sampling_frequency % 2 == 0:
        resampling_frequency = 256
    else:
        resampling_frequency = 255

    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    resampling_frequency = sampling_frequency * up / down
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data

    return data, resampling_frequency

# Extract features.
def get_features(data_folder, patient_id):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids = find_recording_files(data_folder, patient_id)
    num_recordings = len(recording_ids)

    # Extract patient features.

    patient_features = get_patient_features(patient_metadata)

    # Extract EEG features.
    eeg_channels = ['F3', 'P3', 'F4', 'P4']

    #eeg_channels = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'O2',
    #                'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'Fz', 'Cz', 'Pz', 'Fpz']
    group = 'EEG'

    if num_recordings > 0:
        all_data = []
        # Find the max recordings
        sampling_frequency = None
        print(time.asctime(), ": Begin extraction of eeg_features for", recording_ids)
        for idx, recording_id in enumerate(recording_ids):
            recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
            if os.path.exists(recording_location + '.hea'):
                data, channels, sampling_frequency = load_recording_data(recording_location)
                utility_frequency = get_utility_frequency(recording_location + '.hea')
                data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                if all(channel in channels for channel in eeg_channels):
                    data, channels = reduce_channels(data, channels, eeg_channels)
                    data = np.array([data[0, :] - data[1, :], data[2, :] - data[3, :]])
                all_data.append(data)
        
        if len(all_data) > 0:
            def min_width(data_list):
                ret = np.inf
                for data in data_list:
                    if data.shape[1] < ret:
                        ret = data.shape[1]
                return ret
            
            def resize(data_list, rows, cols):
                ret = []
                for data in data_list:
                    ret.append(np.resize(data, (rows, cols)))
                return ret

            width = min_width(all_data)
            all_data = resize(all_data, all_data[0].shape[0], width)
            data = np.dstack(all_data)
            eeg_features = extract_eeg_features(data, sampling_frequency).flatten()
            print(time.asctime(), ": eeg_features for", recording_ids, ": shape=", eeg_features.shape)
        else:
            print(patient_id, "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            eeg_features = float('nan') * np.ones(40) # 2 bipolar channels * 20 features / channel
    else:
        print(patient_id, "YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
        eeg_features = float('nan') * np.ones(40) # 2 bipolar channels * 20 features / channel
    

    # Extract ECG features.
    ecg_channels = ['ECG', 'ECGL', 'ECGR', 'ECG1', 'ECG2']
    group = 'ECG'

    if num_recordings > 0:
        recording_id = recording_ids[0]
        recording_location = os.path.join(data_folder, patient_id, '{}_{}'.format(recording_id, group))
        if os.path.exists(recording_location + '.hea'):
            print('recording id is', recording_id)
            data, channels, sampling_frequency = load_recording_data(recording_location)
            utility_frequency = get_utility_frequency(recording_location + '.hea')

            data, channels = reduce_channels(data, channels, ecg_channels)
            data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
            features = get_ecg_features(data)
            ecg_features = expand_channels(features, channels, ecg_channels).flatten()
        else:
            print(patient_id, "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")
            ecg_features = float('nan') * np.ones(10) # 5 channels * 2 features / channel
    else:
        print(patient_id, "WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
        ecg_features = float('nan') * np.ones(10) # 5 channels * 2 features / channel

    # Extract features.
    return np.hstack((patient_features, eeg_features, ecg_features))

# Extract patient features from the data.
def get_patient_features(data):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1

    features = np.array((age, female, male, other, rosc, ohca, shockable_rhythm, ttm))

    return features

def extract_eeg_features(data, sampling_frequency):
    if data.ndim < 2 or data.ndim > 3:
        raise ValueError('Data dimension not supported')
    if data.ndim == 2:
        data = data[..., np.newaxis]
    return eegf.extract_all_eeg_features(data, sampling_frequency) 

# Extract features from the EEG data.
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=0.5,  fmax=8.0, verbose=False)
        theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=4.0,  fmax=8.0, verbose=False)
        alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency,  fmin=8.0, fmax=12.0, verbose=False)
        beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False)

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean  = np.nanmean(beta_psd,  axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float('nan') * np.ones(num_channels)

    features = np.array((delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)).T

    return features

# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)
    elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std  = float('nan') * np.ones(num_channels)
    else:
        mean = float('nan') * np.ones(num_channels)
        std = float('nan') * np.ones(num_channels)

    features = np.array((mean, std)).T

    return features
