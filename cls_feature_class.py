# Contains routines for labels creation, features extraction and normalization
#


import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
import joblib
from IPython import embed
import matplotlib.pyplot as plot
import librosa

from kymatio.scattering1d.utils import compute_meta_scattering
import torch

plot.switch_backend('agg')
import shutil
import math


def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n - r)


class FeatureClass:
    def __init__(self, params, is_eval=False):
        """

        :param params: parameters dictionary
        :param is_eval: if True, does not load dataset labels.
        """

        # Input directories
        self._feat_label_dir = params['feat_label_dir']
        self._dataset_dir = params['dataset_dir']
        self._dataset_combination = '{}_{}'.format(params['dataset'], 'eval' if is_eval else 'dev')
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)

        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')

        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None

        # Local parameters
        self._is_eval = is_eval

        self._fs = params['fs']
        self._hop_len_s = params['hop_len_s']
        self._hop_len = int(self._fs * self._hop_len_s)

        self._label_hop_len_s = params['label_hop_len_s']
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)
        self._nb_mel_bins = params['nb_mel_bins']
        self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T

        self._dataset = params['dataset']
        self._eps = 1e-8
        self._nb_channels = 4

        self._wave_let = params['wavelet_family']  # for discrete wavelet transform
        self._wave_let_mode = params['wavelet_mode']  # for discrete wavelet transform
        self._scatter_wavelet_Q = params['scatter_wavelet_Q']  # for scatter wavelet transform
        self._scatter_wavelet_J = params['scatter_wavelet_J']  # for scatter wavelet transform
        self._cwave_let = params['cwavelet_family']  # for continuous wavelet transform
        self._cwave_let_mode = params['cwavelet_mode']  # for continuous wavelet transform
        self._is_mel = params['is_mel']
        self._scaler_type = params['scaler_type']

        # Sound event classes dictionary
        self._unique_classes = params['unique_classes']
        self._audio_max_len_samples = params[
                                          'max_audio_len_s'] * self._fs  # TODO: Fix the audio synthesis code to always generate 60s of
        # audio. Currently, it generates audio till the last active sound event, which is not always 60s long. This is a
        # quick fix to overcome that. We need this because, for processing and training we need the length of features
        # to be fixed.

        self._max_feat_frames = int(np.ceil(self._audio_max_len_samples / float(self._hop_len)))
        self._max_label_frames = int(np.ceil(self._audio_max_len_samples / float(self._label_hop_len)))

        self._meta = compute_meta_scattering(self._scatter_wavelet_J, self._scatter_wavelet_Q)  # to compute
        # scattering wavelet params
        self._order0_indices = (self._meta['order'] == 0)  # scattering wavelet order zero indices
        self._order1_indices = (self._meta['order'] == 1)  # scattering wavelet order one indices
        self._order2_indices = (self._meta['order'] == 2)  # scattering wavelet order two indices
        self._order21_indices = (self._meta['order'] != 0)  # scattering wavelet order one,two indices
        self._feature_list = params['feature_list']  # list of features to be extracted

    def _load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)
        audio = audio[:, :self._nb_channels] / 32768.0 + self._eps
        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = np.random.rand(self._audio_max_len_samples - audio.shape[0], audio.shape[1]) * self._eps
            audio = np.vstack((audio, zero_pad))
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :]
        return audio, fs

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def _spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        spectra = np.zeros((self._max_feat_frames, nb_bins + 1, _nb_ch), dtype=complex)
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft,
                                        hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra[:, :, ch_cnt] = stft_ch[:, :self._max_feat_frames].T
        return spectra

    def _get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt]) ** 2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        # mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))   #avoid this part to have
        # a 3D array
        return mel_feat

    def _get_foa_intensity_vectors(self, linear_spectra):
        IVx = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 3])
        IVy = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 1])
        IVz = np.real(np.conj(linear_spectra[:, :, 0]) * linear_spectra[:, :, 2])

        normal = self._eps + (np.abs(linear_spectra[:, :, 0]) ** 2 + np.abs(linear_spectra[:, :, 1]) ** 2 + np.abs(
            linear_spectra[:, :, 2]) ** 2 + np.abs(linear_spectra[:, :, 3]) ** 2) / 2.
        # normal = np.sqrt(IVx**2 + IVy**2 + IVz**2) + self._eps
        # first extract non-mel IV here and then extract IV-mel
        IVxx = IVx / normal
        IVyy = IVy / normal
        IVzz = IVz / normal
        IVx = np.dot(IVx / normal, self._mel_wts)
        IVy = np.dot(IVy / normal, self._mel_wts)
        IVz = np.dot(IVz / normal, self._mel_wts)

        # we are doing the following instead of simply concatenating to keep the processing similar to mel_spec and gcc
        foa_iv = np.dstack((IVx, IVy, IVz))
        foa_ivv = np.dstack((IVyy, IVzz, IVxx))
        # foa_iv = foa_iv.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))   #avoid this part to have
        # a 3D array
        if np.isnan(foa_iv).any():
            print('Feature extraction is generating nan outputs')
            exit()
        return foa_iv, foa_ivv

    # won't be needing this, in case of use, pay attention that this feature is 2D and the rest of features are 3D
    def _get_gcc(self, linear_spectra):
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m + 1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j * np.angle(R)))
                cc = np.concatenate((cc[:, -self._nb_mel_bins // 2:], cc[:, :self._nb_mel_bins // 2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    def _get_spectrogram_for_file(self, audio_path):
        audio_in, fs = self._load_audio(audio_path)
        audio_spec = self._spectrogram(audio_in)
        return audio_spec

    # OUTPUT LABELS
    def get_labels_for_file(self, _desc_file):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: metadata description file
        :return: label_mat: labels of the format [sed_label, doa_label],
        where sed_label is of dimension [nb_frames, nb_classes] which is 1 for active sound event else zero
        where doa_labels is of dimension [nb_frames, 3*nb_classes], nb_classes each for x, y, z axis,
        """

        se_label = np.zeros((self._max_label_frames, len(self._unique_classes)))
        x_label = np.zeros((self._max_label_frames, len(self._unique_classes)))
        y_label = np.zeros((self._max_label_frames, len(self._unique_classes)))
        z_label = np.zeros((self._max_label_frames, len(self._unique_classes)))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < self._max_label_frames:
                for active_event in active_event_list:
                    se_label[frame_ind, active_event[0]] = 1
                    x_label[frame_ind, active_event[0]] = active_event[1]
                    y_label[frame_ind, active_event[0]] = active_event[2]
                    z_label[frame_ind, active_event[0]] = active_event[3]

        label_mat = np.concatenate((se_label, x_label, y_label, z_label), axis=1)
        return label_mat

    def costume_padding(self, arr, pad_size):
        # this is to do some padding so all the extracted features have the same dimension (excluding the channel
        # dimension) tweaking to fit IPD, since IPD has 513 features and first order has 512
        diff = pad_size - arr.shape[0]
        ol_shape = list(arr.shape)
        ol_shape[0] = pad_size
        # ol_shape[1] = ol_shape[1] + 1
        out = np.zeros(ol_shape)
        # indices = [i for i in range(pad_size) if i % (pad_size // diff) == pad_size // diff - 1]
        indices = [i for i in range(pad_size // diff * diff) if i % (pad_size // diff) == pad_size // diff - 1]
        # ex_indices = [i for i in range(pad_size) if i % (pad_size // diff) != pad_size // diff - 1]
        ex_indices = [i for i in range(pad_size) if i not in indices]
        # out[indices, :, :] = 0
        # out[ex_indices, 1:, :] = arr
        out[ex_indices, :, :] = arr
        for i in indices:
            if i < pad_size - 1:
                # out[i, 1:, :] = (out[i + 1, 1:, :] + out[i - 1, 1:, :]) / 2
                out[i, :, :] = (out[i + 1, :, :] + out[i - 1, :, :]) / 2
        return out

    def costume_downsampling(self, arr, pad_size):
        # in case a feature has larger dimensions, this functions down-samples it
        # tweaking to fit IPD, since IPD has 513 features and first order has 512
        diff = arr.shape[1] - arr.shape[1] // pad_size * pad_size
        ol_shape = list(arr.shape)
        ol_shape[1] = pad_size
        out = np.zeros(ol_shape)
        # indices = [i for i in range(pad_size) if i % (pad_size // diff) == pad_size // diff - 1]
        indices = [i + diff // 2 for i in range(arr.shape[1] // diff * diff) if
                   i % (arr.shape[1] // diff) == arr.shape[1] // diff - 1]
        # ex_indices = [i for i in range(pad_size) if i % (pad_size // diff) != pad_size // diff - 1]
        ex_indices = [i for i in range(arr.shape[1]) if i not in indices]
        # out[indices, :, :] = 0
        out = arr[:, ex_indices, :]
        out = torch.nn.AvgPool2d((arr.shape[1] // pad_size, 1), stride=(arr.shape[1] // pad_size, 1))(torch.tensor(out))
        return out

    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------
    def extract_all_feature(self):
        # this is to extract the original mel-spectrogram and mel-IV features
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir()
        create_folder(self._feat_dir)

        # extraction starts
        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))
        for split in os.listdir(self._aud_dir):
            print('Split: {}'.format(split))
            for file_cnt, file_name in enumerate(os.listdir(os.path.join(self._aud_dir, split))):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                spect = self._get_spectrogram_for_file(os.path.join(self._aud_dir, split, wav_filename))

                # extract mel
                mel_spect = self._get_mel_spectrogram(spect)

                feat = None
                if self._dataset is 'foa':
                    # extract intensity vectors
                    foa_iv, _ = self._get_foa_intensity_vectors(spect)
                    feat = np.concatenate((mel_spect, foa_iv), axis=-1)
                elif self._dataset is 'mic':
                    # extract gcc
                    gcc = self._get_gcc(spect)
                    feat = np.concatenate((mel_spect, gcc), axis=-1)
                else:
                    print('ERROR: Unknown dataset format {}'.format(self._dataset))
                    exit()

                if feat is not None:
                    # the Mel-Spectrogram and Mel-IV features are concatenated and preprocessed together in the next
                    # function
                    print('\t{}: {}, {}'.format(file_cnt, file_name, feat.shape))
                    np.save(os.path.join(self._feat_dir, '{}.npy'.format(wav_filename.split('.')[0])), feat)

    def preprocess_features(self):
        # preprocess concatenated Mel-Spectrogram and Mel-IV based on original code
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()
        spec_scaler = None

        # pre-processing starts
        if self._is_eval:
            spec_scaler = joblib.load(normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Loaded.'.format(normalized_features_wts_file))

        else:
            print('Estimating weights for normalizing feature files:')
            print('\t\tfeat_dir: {}'.format(self._feat_dir))

            spec_scaler = preprocessing.StandardScaler()
            for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print('{}: {}'.format(file_cnt, file_name))
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                spec_scaler.partial_fit(feat_file.transpose((0, 2, 1)).reshape(self._max_feat_frames, -1))  # do the
                # transpose to convert 3D feature to 2D for standardization
                del feat_file
            joblib.dump(
                spec_scaler,
                normalized_features_wts_file
            )
            print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

        print('Normalizing feature files:')
        print('\t\tfeat_dir_norm {}'.format(self._feat_dir_norm))
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self._feat_dir, file_name))
            feat_file_shape = feat_file.shape
            feat_file = spec_scaler.transform(
                feat_file.transpose((0, 2, 1)).reshape(self._max_feat_frames, -1)).reshape(
                self._max_feat_frames, feat_file_shape[-1], -1).transpose((0, 2, 1))  # after standardization, we change
            # the shape to 3D
            np.save(
                os.path.join(self._feat_dir_norm, file_name),
                feat_file
            )
            del feat_file

        print('normalized files written to {}'.format(self._feat_dir_norm))

    # -------------------------- Extract Desired features and Preprocess them ------------------------

    def extract_spectrogram_etc(self):
        # here we extract spectrogram, IPD, and IV without MEL based on self._feature_list
        self._feat_dir = self.get_unnormalized_feat_dir()
        for feat_name in ['_spec', '_IV', '_IPD', '_IPD_Cos', 'IPD_Sin']:
            if feat_name in self._feature_list:
                create_folder(self._feat_dir + feat_name)
        print(f'Extracting {feat_name}:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))
        for split in os.listdir(self._aud_dir):
            print('Split: {}'.format(split))
            for file_cnt, file_name in enumerate(os.listdir(os.path.join(self._aud_dir, split))):
                print('\t{}: {}'.format(file_cnt, file_name))
                # if file_cnt == 2:
                #     break
                spect = None
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                spect = self._get_spectrogram_for_file(os.path.join(self._aud_dir, split, wav_filename))
                for feat_name in ['_spec', '_IV', '_IPD', '_IPD_Cos', 'IPD_Sin']:
                    if feat_name in self._feature_list:
                        if feat_name == '_spec':
                            log_spect = librosa.power_to_db((np.abs(spect)) ** 2)
                            np.save(os.path.join(self._feat_dir + '_spec', '{}.npy'.format(wav_filename.split('.')[0])),
                                    log_spect)
                        elif feat_name == '_IV':
                            _, foa_iv = self._get_foa_intensity_vectors(spect)
                            np.save(os.path.join(self._feat_dir + '_IV', '{}.npy'.format(wav_filename.split('.')[0])),
                                    foa_iv)
                        else:
                            phase_vector = np.angle(spect[..., 1:] * np.conj(spect[:, :, 0, None]))
                            if feat_name == '_IPD':
                                np.save(
                                    os.path.join(self._feat_dir + '_IPD', '{}.npy'.format(wav_filename.split('.')[0])),
                                    phase_vector / np.pi)
                            elif feat_name == '_IPD_Cos':
                                np.save(os.path.join(self._feat_dir + '_IPD_Cos',
                                                     '{}.npy'.format(wav_filename.split('.')[0])),
                                        np.cos(phase_vector))
                            elif feat_name == 'IPD_Sin':
                                np.save(os.path.join(self._feat_dir + '_IPD_Sin',
                                                     '{}.npy'.format(wav_filename.split('.')[0])),
                                        np.sin(phase_vector))

    def preprocess_spec_etc(self):
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()
        for feat_name in ['_spec', '_IV', '_IPD', '_IPD_Cos', 'IPD_Sin']:
            if feat_name in self._feature_list:
                create_folder(self._feat_dir_norm + feat_name)
                for scaler_type in self._scaler_type:
                    feat_wts = self.get_normalized_wts_file() + scaler_type + feat_name
                    if self._is_eval:
                        feat_scaler = joblib.load(feat_wts)
                    else:
                        feat_scaler = preprocessing.MinMaxScaler() if scaler_type == 'minmax' else preprocessing.StandardScaler()
                        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir + feat_name)):
                            print('{}: {}'.format(file_cnt, file_name))
                            feat = np.load(os.path.join(self._feat_dir + feat_name, file_name))
                            feat_scaler.partial_fit(feat.transpose((0, 2, 1)).reshape(self._max_feat_frames, -1))
                        joblib.dump(feat_scaler, feat_wts)
                        del feat
                    for file_cnt, file_name in enumerate(os.listdir(self._feat_dir + feat_name)):
                        print('{}: {}'.format(file_cnt, file_name))
                        feat = np.load(os.path.join(self._feat_dir + feat_name, file_name))
                        nb_ch = feat.shape[-1]
                        feat_scl = feat_scaler.transform(feat.transpose((0, 2, 1)).reshape(self._max_feat_frames, -1)).reshape(
                            self._max_feat_frames, nb_ch, -1).transpose((0, 2, 1))
                        np.save(os.path.join(self._feat_dir_norm + feat_name, file_name), feat_scl)
                    del feat
                    del feat_scl






    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self._label_dir = self.get_label_dir()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        create_folder(self._label_dir)
        for split in os.listdir(self._desc_dir):
            print('Split: {}'.format(split))
            for file_cnt, file_name in enumerate(os.listdir(os.path.join(self._desc_dir, split))):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                desc_file_polar = self.load_output_format_file(os.path.join(self._desc_dir, split, file_name))
                desc_file = self.convert_output_format_polar_to_cartesian(desc_file_polar)
                label_mat = self.get_labels_for_file(desc_file)
                print('\t{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
                np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

    # -------------------------------  DCASE OUTPUT  FORMAT FUNCTIONS -------------------------------
    def load_output_format_file(self, _output_format_file):
        """
        Loads DCASE output format csv file and returns it in dictionary format

        :param _output_format_file: DCASE output format CSV
        :return: _output_dict: dictionary
        """
        _output_dict = {}
        _fid = open(_output_format_file, 'r')
        # next(_fid)
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 5:  # read polar coordinates format, we ignore the track count
                _output_dict[_frame_ind].append([int(_words[1]), float(_words[3]), float(_words[4]), int(_words[2])])
            elif len(_words) == 6:  # read Cartesian coordinates format, we ignore the track count
                _output_dict[_frame_ind].append(
                    [int(_words[1]), float(_words[3]), float(_words[4]), float(_words[5]), int(_words[2])])
        _fid.close()
        return _output_dict

    def write_output_format_file(self, _output_format_file, _output_format_dict):
        """
        Writes DCASE output format csv file, given output format dictionary

        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count we use a fixed value.
                _fid.write(
                    '{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]),
                                                 float(_value[3])))
        _fid.close()

    def segment_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in segments of length 1s from reference dataset
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each segment of audio
                dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
        '''
        nb_blocks = int(np.ceil(_max_frames / float(self._nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}
        for frame_cnt in range(0, _max_frames, self._nb_label_frames_1s):

            # Collect class-wise information for each block
            # [class][frame] = <list of doa values>
            # Data structure supports multi-instance occurence of same class
            block_cnt = frame_cnt // self._nb_label_frames_1s
            loc_dict = {}
            for audio_frame in range(frame_cnt, frame_cnt + self._nb_label_frames_1s):
                if audio_frame not in _pred_dict:
                    continue
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict:
                        loc_dict[value[0]] = {}

                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]:
                        loc_dict[value[0]][block_frame] = []
                    loc_dict[value[0]][block_frame].append(value[1:])

            # Update the block wise details collected above in a global structure
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_cnt] = []

                keys = [k for k in loc_dict[class_cnt]]
                values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

                output_dict[block_cnt][class_cnt].append([keys, values])

        return output_dict

    def regression_label_format_to_output_format(self, _sed_labels, _doa_labels):
        """
        Converts the sed (classification) and doa labels predicted in regression format to dcase output format.

        :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
        :param _doa_labels: DOA labels matrix [nb_frames, 2*nb_classes] or [nb_frames, 3*nb_classes]
        :return: _output_dict: returns a dict containing dcase output format
        """

        _nb_classes = len(self._unique_classes)
        _is_polar = _doa_labels.shape[-1] == 2 * _nb_classes
        _azi_labels, _ele_labels = None, None
        _x, _y, _z = None, None, None
        if _is_polar:
            _azi_labels = _doa_labels[:, :_nb_classes]
            _ele_labels = _doa_labels[:, _nb_classes:]
        else:
            _x = _doa_labels[:, :_nb_classes]
            _y = _doa_labels[:, _nb_classes:2 * _nb_classes]
            _z = _doa_labels[:, 2 * _nb_classes:]

        _output_dict = {}
        for _frame_ind in range(_sed_labels.shape[0]):
            _tmp_ind = np.where(_sed_labels[_frame_ind, :])
            if len(_tmp_ind[0]):
                _output_dict[_frame_ind] = []
                for _tmp_class in _tmp_ind[0]:
                    if _is_polar:
                        _output_dict[_frame_ind].append(
                            [_tmp_class, _azi_labels[_frame_ind, _tmp_class], _ele_labels[_frame_ind, _tmp_class]])
                    else:
                        _output_dict[_frame_ind].append(
                            [_tmp_class, _x[_frame_ind, _tmp_class], _y[_frame_ind, _tmp_class],
                             _z[_frame_ind, _tmp_class]])
        return _output_dict

    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    ele_rad = tmp_val[2] * np.pi / 180.
                    azi_rad = tmp_val[1] * np.pi / 180

                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append([tmp_val[0], x, y, z, tmp_val[-1]])
        return out_dict

    def convert_output_format_cartesian_to_polar(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    x, y, z = tmp_val[1], tmp_val[2], tmp_val[3]

                    # in degrees
                    azimuth = np.arctan2(y, x) * 180 / np.pi
                    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi
                    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                    out_dict[frame_cnt].append([tmp_val[0], azimuth, elevation, tmp_val[-1]])
        return out_dict

    # ------------------------------- Misc public functions -------------------------------
    def get_classes(self):
        return self._unique_classes

    def get_normalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_norm'.format(self._dataset_combination)
        )

    def get_unnormalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}'.format(self._dataset_combination)
        )

    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            return os.path.join(
                self._feat_label_dir, '{}_label'.format(self._dataset_combination)
            )

    def get_normalized_wts_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_wts'.format(self._dataset)
        )

    def get_nb_channels(self):
        return self._nb_channels

    def get_nb_classes(self):
        return len(self._unique_classes)

    def nb_frames_1s(self):
        return self._nb_label_frames_1s

    def get_hop_len_sec(self):
        return self._hop_len_s

    def get_nb_frames(self):
        return self._max_label_frames

    def get_nb_mel_bins(self):
        return self._nb_mel_bins


def create_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)


def delete_and_create_folder(folder_name):
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)
