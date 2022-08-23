#
# Data generator for training the SELDnet. Here we add some extra features such as online data augmentation. also
# make the code compatible with the customized clc_feature_class with added features
#

import os
import numpy as np
import cls_feature_class
from IPython import embed
from collections import deque
import random

import torch


def costume_padding(arr, pad_size):
    # tweaking to fit IPD, since IPD has 513 features and first order has 512
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


def costume_downsampling(arr, pad_size):
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


class DataGenerator(object):
    def __init__(
            self, params, split=1, shuffle=True, per_file=False, is_eval=False
    ):
        self._per_file = per_file
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._batch_size = params['batch_size']

        self._scatter_wavelet_Q = params['scatter_wavelet_Q']

        self._feature_seq_len = params['feature_sequence_length']
        self._label_seq_len = params['label_sequence_length']
        self._is_accdoa = params['is_accdoa']
        self._doa_objective = params['doa_objective']
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()

        self._feat_list = params['feature_list']
        self._is_mel = params['is_mel']
        self._scaler_type = params['scaler_type']
        self._is_online_aug = params['is_online_aug']
        self._is_acs_aug = params['is_acs_aug']

        self._filenames_list = list()
        self._nb_frames_file = 0  # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins() if self._is_mel else self._feat_cls._nfft // 2 + 1  # todo
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None  # DOA label length
        self._class_dict = self._feat_cls.get_classes()
        self._nb_classes = self._feat_cls.get_nb_classes()
        self._get_filenames_list_and_feat_label_sizes()

        self._feature_batch_seq_len = self._batch_size * self._feature_seq_len
        self._label_batch_seq_len = self._batch_size * self._label_seq_len
        self._circ_buf_feat = None
        self._circ_buf_label = None

        if self._per_file:
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor((len(self._filenames_list) * self._nb_frames_file /
                                                   float(self._feature_batch_seq_len))))

        # self._dummy_feat_vec = np.ones(self._feat_len.shape) *

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list), self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch, self._label_len
            )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                params['dataset'], split,
                self._batch_size, self._feature_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir
            )
        )

    def get_data_sizes(self):
        if self._is_mel:
            feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._nb_mel_bins)
        else:
            # Can be more general but we usually go and make all the features the same size as our spectrogram
            feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._feat_cls._nfft // 2 + 1)
        if self._is_eval:
            label_shape = None
        else:
            if self._is_accdoa:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes * 3)
            else:
                label_shape = [
                    (self._batch_size, self._label_seq_len, self._nb_classes),
                    (self._batch_size, self._label_seq_len, self._nb_classes * 3)
                ]
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def _get_filenames_list_and_feat_label_sizes(self):
        if self._is_mel:
            for filename in os.listdir(
                    self._feat_dir + f'_mel{self._nb_mel_bins}_' + self._scaler_type + self._feat_list[0]):
                if self._is_eval:
                    self._filenames_list.append(filename)
                else:
                    if int(filename[4]) in self._splits:  # check which split the file belongs to
                        self._filenames_list.append(filename)

            temp_feat = np.load(
                os.path.join(self._feat_dir + f'_mel{self._nb_mel_bins}_' + self._scaler_type + self._feat_list[0],
                             self._filenames_list[0]))
            for feat_num, feat_name in enumerate(self._feat_list):
                if feat_num == 0:
                    continue
                temp_feat = np.concatenate((temp_feat, np.load(
                    os.path.join(self._feat_dir + f'_mel{self._nb_mel_bins}_' + self._scaler_type + feat_name,
                                 self._filenames_list[0]))), axis=-1)

            self._nb_frames_file = temp_feat.shape[0]
            self._nb_ch = temp_feat.shape[-1]

            if not self._is_eval:
                temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
                self._label_len = temp_label.shape[-1]
                self._doa_len = (self._label_len - self._nb_classes) // self._nb_classes

            if self._per_file:
                self._batch_size = int(np.ceil(temp_feat.shape[0] / float(self._feature_seq_len)))
            del temp_feat
        else:
            for filename in os.listdir(
                    self._feat_dir + '_' + self._scaler_type + self._feat_list[0]):
                if self._is_eval:
                    self._filenames_list.append(filename)
                else:
                    if int(filename[4]) in self._splits:  # check which split the file belongs to
                        self._filenames_list.append(filename)
            temp_feat = np.load(
                os.path.join(self._feat_dir + '_' + self._scaler_type + self._feat_list[0],
                             self._filenames_list[0]))
            if temp_feat.shape[0] < self._feat_cls._max_feat_frames:
                temp_feat = costume_padding(temp_feat, self._feat_cls._max_feat_frames)

            if temp_feat.shape[1] <= (self._feat_cls._nfft // 2 + 1):
                temp_feat = np.concatenate((temp_feat, np.zeros(
                    (temp_feat.shape[0], (self._feat_cls._nfft // 2 + 1) - temp_feat.shape[1], temp_feat.shape[-1]))),
                                           axis=1)
                # feat[:, :temp_feat.shape[1], :] = temp_feat
            else:
                temp_feat = costume_downsampling(temp_feat, self._feat_cls._nfft // 2 + 1)
            for feat_num, feat_name in enumerate(self._feat_list):
                if feat_num == 0:
                    continue

                feat = np.load(
                    os.path.join(self._feat_dir + '_' + self._scaler_type + self._feat_list[0],
                                 self._filenames_list[0]))
                if feat.shape[0] < self._feat_cls._max_feat_frames:
                    feat = costume_padding(feat, self._feat_cls._max_feat_frames)

                if feat.shape[1] <= (self._feat_cls._nfft // 2 + 1):
                    feat = np.concatenate((feat, np.zeros(
                        (feat.shape[0], (self._feat_cls._nfft // 2 + 1) - feat.shape[1],
                         feat.shape[-1]))),
                                          axis=1)
                    # feat[:, :temp_feat.shape[1], :] = temp_feat
                else:
                    feat = costume_downsampling(feat, self._feat_cls._nfft // 2 + 1)

                temp_feat = np.concatenate((temp_feat, feat), axis=-1)

            self._nb_frames_file = temp_feat.shape[0]
            self._nb_ch = temp_feat.shape[-1]

            if not self._is_eval:
                temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
                self._label_len = temp_label.shape[-1]
                self._doa_len = (self._label_len - self._nb_classes) // self._nb_classes

            if self._per_file:
                self._batch_size = int(np.ceil(temp_feat.shape[0] / float(self._feature_seq_len)))
            del temp_feat
            del feat

        return

    def RandomCutoutHoleNp(self, x, always_apply: bool = False, p: float = 0.5, n_max_holes: int = 8,
                           max_h_size: int = 8,
                           max_w_size: int = 8, filled_value: float = None, n_zero_channels: int = None,
                           is_filled_last_channels: bool = True):

        """
                :param always_apply: If True, always apply transform.
                :param p: If always_apply is false, p is the probability to apply transform.
                :param n_max_holes: Maximum number of holes to cutout.
                :param max_h_size: Maximum time frames of the cutout holes.
                :param max_w_size: Maximum freq bands of the cutout holes.
                :param filled_value: random value to fill in the cutout area. If None, randomly fill the cutout area with value
                    between min and max of input.
                :param n_zero_channels: if given, these last n_zero_channels will be filled in zeros instead of random values
                :param is_filled_last_channels: if False, does not cutout n_zero_channels
                :param x: is the input
                """
        self.n_max_holes = n_max_holes
        self.max_h_size = np.max((max_h_size, 5))
        self.max_w_size = np.max((max_w_size, 5))
        self.filled_value = filled_value
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

        if always_apply:
            rand_num = 0
        else:
            rand_num = np.random.rand()
        if rand_num < p:
            # print('Applying RandomCutOutHoles')
            assert x.ndim == 3, 'Error: dimension of input spectrogram is not 3!'
            img_h = x.shape[0]  # time frame dimension
            img_w = x.shape[1]  # feature dimension
            min_value = np.min(x)
            max_value = np.max(x)
            new_spec = x.copy()
            # n_cutout_holes = np.random.randint(1, self.n_max_holes, 1)[0]
            n_cutout_holes = self.n_max_holes
            for ihole in np.arange(n_cutout_holes):
                # w = np.random.randint(4, self.max_w_size, 1)[0]
                # h = np.random.randint(4, self.max_h_size, 1)[0]
                w = self.max_w_size
                h = self.max_h_size
                left = np.random.randint(0, img_w - w)
                top = np.random.randint(0, img_h - h)
                if self.filled_value is None:
                    filled_value = np.random.uniform(min_value, max_value)
                else:
                    filled_value = self.filled_value
                if self.n_zero_channels is None:
                    new_spec[top:top + h, left:left + w, :] = filled_value
                else:
                    new_spec[top:top + h, left:left + w, :-self.n_zero_channels] = filled_value
                    if self.is_filled_last_channels:
                        new_spec[top:top + h, left:left + w, -self.n_zero_channels:] = 0.0

            return new_spec
        else:
            print('Applying Nothing')
            return x

    def SpecAugmentNp(self, x, always_apply: bool = False, p: float = 0.5, time_max_width: int = None,
                      freq_max_width: int = None, n_time_stripes: int = 1, n_freq_stripes: int = 1,
                      n_zero_channels: int = None, is_filled_last_channels: bool = True):

        """

        :param always_apply: If True, always apply transform. :param p: If always_apply is false,
        p is the probability to apply transform. :param time_max_width: maximum time width to remove. :param
        freq_max_width: maximum freq width to remove. :param n_time_stripes: number of time stripes to remove. :param
        n_freq_stripes: number of freq stripes to remove. :param n_zero_channels: if given, these last
        n_zero_channels will be filled in zeros instead of random values :param is_filled_last_channels: if False,
        does not cutout n_zero_channels :param x: is the input

        """
        self.time_max_width = time_max_width
        self.freq_max_width = freq_max_width
        self.n_time_stripes = n_time_stripes
        self.n_freq_stripes = n_freq_stripes
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels

        if always_apply:
            rand_num = 0
        else:
            rand_num = np.random.rand()
        if rand_num < p:
            # print('Applying SpecAugment')
            assert x.ndim == 3, 'Error: dimension of input spectrogram is not 3!'
            n_frames = x.shape[0]
            n_freqs = x.shape[1]
            min_value = np.min(x)
            max_value = np.max(x)
            if self.time_max_width is None:
                time_max_width = int(0.15 * n_frames)
            else:
                time_max_width = self.time_max_width
            time_max_width = np.max((1, time_max_width))
            if self.freq_max_width is None:
                freq_max_width = int(0.2 * n_freqs)
            else:
                freq_max_width = self.freq_max_width
            freq_max_width = np.max((1, freq_max_width))

            new_spec = x.copy()

            for i in np.arange(self.n_time_stripes):
                dur = np.random.randint(1, time_max_width, 1)[0]
                start_idx = np.random.randint(0, n_frames - dur, 1)[0]
                random_value = np.random.uniform(min_value, max_value, 1)
                if self.n_zero_channels is None:
                    new_spec[start_idx:start_idx + dur, :, :] = random_value
                else:
                    new_spec[start_idx:start_idx + dur, :, :-self.n_zero_channels] = random_value
                    if self.is_filled_last_channels:
                        new_spec[start_idx:start_idx + dur, :, -self.n_zero_channels:, ] = 0.0

            for i in np.arange(self.n_freq_stripes):
                dur = np.random.randint(1, freq_max_width, 1)[0]
                start_idx = np.random.randint(0, n_freqs - dur, 1)[0]
                random_value = np.random.uniform(min_value, max_value, 1)
                if self.n_zero_channels is None:
                    new_spec[:, start_idx:start_idx + dur, :] = random_value
                else:
                    new_spec[:, start_idx:start_idx + dur, :-self.n_zero_channels] = random_value
                    if self.is_filled_last_channels:
                        new_spec[:, start_idx:start_idx + dur, -self.n_zero_channels:] = 0.0

            return new_spec
        else:
            # print('Applying Nothing')
            return x

    def RandomShiftUpDownNp(self, x, always_apply=False, p=0.5, freq_shift_range: int = None, direction: str = None,
                            mode='reflect',
                            n_last_channels: int = 0):

        self.freq_shift_range = freq_shift_range
        self.direction = direction
        self.mode = mode
        self.n_last_channels = n_last_channels
        if always_apply:
            rand_num = 0
        else:
            rand_num = np.random.rand()
        if rand_num < p:
            # print('Applying RandomShiftUpDown')
            n_timesteps, n_features, n_channels = x.shape
            if self.freq_shift_range is None:
                self.freq_shift_range = int(n_features * 0.08)
            shift_len = np.random.randint(1, self.freq_shift_range, 1)[0]
            if self.direction is None:
                direction = np.random.choice(['up', 'down'], 1)[0]
            else:
                direction = self.direction
            new_spec = x.copy()
            if self.n_last_channels == 0:
                if direction == 'up':
                    new_spec = np.pad(new_spec, ((0, 0), (shift_len, 0), (0, 0)), mode=self.mode)[:, 0:n_features,
                               :]
                else:
                    new_spec = np.pad(new_spec, ((0, 0), (0, shift_len), (0, 0)), mode=self.mode)[:, shift_len:, :]
            else:
                if direction == 'up':
                    new_spec[..., :-self.n_last_channels] = np.pad(
                        new_spec[..., :-self.n_last_channels], ((0, 0), (shift_len, 0), (0, 0)), mode=self.mode)[:,
                                                            0:n_features, :]
                else:
                    new_spec[..., :-self.n_last_channels] = np.pad(
                        new_spec[..., :-self.n_last_channels], ((0, 0), (0, shift_len), (0, 0)), mode=self.mode)[:,
                                                            shift_len:, :]
            return new_spec
        else:
            # print('Applying Nothing')
            return x

    def RandomCutoutNp(self, x, always_apply: bool = False, p: float = 0.5, image_aspect_ratio: float = 1,
                       random_value: float = None, n_zero_channels: int = None, is_filled_last_channels: bool = True):
        """
        :param always_apply: If True, always apply transform. :param p: If always_apply is false,
        p is the probability to apply transform. :param image_aspect_ratio: height/width ratio. For spectrogram:
        n_time_steps/ n_features. :param random_value: random value to fill in the cutout area. If None,
        randomly fill the cutout area with value between min and max of input. :param n_zero_channels: if given,
        these last n_zero_channels will be filled in zeros instead of random values :param is_filled_last_channels:
        if False, does not cutout n_zero_channels :param x: is the input
        """
        self.random_value = random_value
        self.n_zero_channels = n_zero_channels
        self.is_filled_last_channels = is_filled_last_channels
        # Params: s: area, r: height/width ratio.
        self.s_l = 0.02
        self.s_h = 0.3
        self.r_1 = 0.3
        self.r_2 = 1 / 0.3
        if image_aspect_ratio > 1:
            self.r_1 = self.r_1 * image_aspect_ratio
        elif image_aspect_ratio < 1:
            self.r_2 = self.r_2 * image_aspect_ratio

        if always_apply:
            rand_num = 0
        else:
            rand_num = np.random.rand()
        if rand_num < p:
            # print('Applying RandomCutOut')
            image_dim = x.ndim
            img_h = x.shape[0]  # time frame dimension
            img_w = x.shape[1]  # feature dimension
            min_value = np.min(x)
            max_value = np.max(x)
            # Initialize output
            output_img = x.copy()
            # random erase
            s = np.random.uniform(self.s_l, self.s_h) * img_h * img_w
            r = np.random.uniform(self.r_1, self.r_2)
            w = np.min((int(np.sqrt(s / r)), img_w - 1))
            h = np.min((int(np.sqrt(s * r)), img_h - 1))
            left = np.random.randint(0, img_w - w)
            top = np.random.randint(0, img_h - h)
            if self.random_value is None:
                c = np.random.uniform(min_value, max_value)
            else:
                c = self.random_value
            if image_dim == 2:
                output_img[top:top + h, left:left + w] = c
            else:
                if self.n_zero_channels is None:
                    output_img[top:top + h, left:left + w, :] = c
                else:
                    output_img[top:top + h, left:left + w, :-self.n_zero_channels] = c
                    if self.is_filled_last_channels:
                        output_img[top:top + h, left:left + w, -self.n_zero_channels:] = 0.0

            return output_img
        else:
            # print('Applying Nothing')
            return x

    def generate(self):
        """
        Generates batches of samples
        :return: 
        """

        while 1:
            if self._shuffle:
                random.shuffle(self._filenames_list)

            # Ideally this should have been outside the while loop. But while generating the test data we want the data
            # to be the same exactly for all epoch's hence we keep it here.
            self._circ_buf_feat = deque()
            self._circ_buf_label = deque()

            file_cnt = 0
            if self._is_eval:
                for i in range(self._nb_total_batches):
                    # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                    # circular buffer. If not keep refilling it.
                    while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                        if self._is_mel:
                            temp_feat = np.load(
                                os.path.join(
                                    self._feat_dir + f'_mel{self._nb_mel_bins}_' + self._scaler_type + self._feat_list[
                                        0],
                                    self._filenames_list[file_cnt]))
                            for feat_num, feat_name in enumerate(self._feat_list):
                                if feat_num == 0:
                                    continue
                                temp_feat = np.concatenate((temp_feat, np.load(
                                    os.path.join(
                                        self._feat_dir + f'_mel{self._nb_mel_bins}_' + self._scaler_type + feat_name,
                                        self._filenames_list[file_cnt]))), axis=-1)
                        else:
                            temp_feat = np.load(
                                os.path.join(self._feat_dir + '_' + self._scaler_type + self._feat_list[0],
                                             self._filenames_list[file_cnt]))
                            if temp_feat.shape[0] < self._feat_cls._max_feat_frames:
                                temp_feat = costume_padding(temp_feat, self._feat_cls._max_feat_frames)

                            if temp_feat.shape[1] <= (self._feat_cls._nfft // 2 + 1):
                                temp_feat = np.concatenate((temp_feat, np.zeros(
                                    (temp_feat.shape[0], (self._feat_cls._nfft // 2 + 1) - temp_feat.shape[1],
                                     temp_feat.shape[-1]))),
                                                           axis=1)
                            else:
                                temp_feat = costume_downsampling(temp_feat, self._feat_cls._nfft // 2 + 1)
                            for feat_num, feat_name in enumerate(self._feat_list):
                                if feat_num == 0:
                                    continue
                                feat = np.load(
                                    os.path.join(self._feat_dir + '_' + self._scaler_type + self._feat_list[0],
                                                 self._filenames_list[file_cnt]))
                                if feat.shape[0] < self._feat_cls._max_feat_frames:
                                    feat = costume_padding(feat, self._feat_cls._max_feat_frames)

                                if feat.shape[1] <= (self._feat_cls._nfft // 2 + 1):
                                    feat = np.concatenate((feat, np.zeros(
                                        (feat.shape[0], (self._feat_cls._nfft // 2 + 1) - feat.shape[1],
                                         feat.shape[-1]))),
                                                          axis=1)
                                    # feat[:, :temp_feat.shape[1], :] = temp_feat
                                else:
                                    feat = costume_downsampling(feat, self._feat_cls._nfft // 2 + 1)

                                temp_feat = np.concatenate((temp_feat, feat), axis=-1)

                        # temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))

                        for row_cnt, row in enumerate(temp_feat):
                            self._circ_buf_feat.append(row)

                        # If self._per_file is True, this returns the sequences belonging to a single audio recording
                        if self._per_file:
                            extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                            extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6

                            for row_cnt, row in enumerate(extra_feat):
                                self._circ_buf_feat.append(row)

                        file_cnt = file_cnt + 1

                    # Read one batch size from the circular buffer
                    feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins, self._nb_ch))
                    for j in range(self._feature_batch_seq_len):
                        feat[j, :] = self._circ_buf_feat.popleft()
                    # feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins)).transpose(
                    #     (0, 2, 1))

                    # Split to sequences
                    feat = self._split_in_seqs(feat, self._feature_seq_len)
                    # feat = np.transpose(feat, (0, 3, 1, 2))

                    yield feat

            else:
                for i in range(self._nb_total_batches):

                    # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                    # circular buffer. If not keep refilling it.
                    while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                        if self._is_mel:
                            temp_feat = np.load(
                                os.path.join(
                                    self._feat_dir + f'_mel{self._nb_mel_bins}_' + self._scaler_type + self._feat_list[
                                        0],
                                    self._filenames_list[file_cnt]))
                            for feat_num, feat_name in enumerate(self._feat_list):
                                if feat_num == 0:
                                    continue
                                temp_feat = np.concatenate((temp_feat, np.load(
                                    os.path.join(
                                        self._feat_dir + f'_mel{self._nb_mel_bins}_' + self._scaler_type + feat_name,
                                        self._filenames_list[file_cnt]))), axis=-1)
                        else:
                            temp_feat = np.load(
                                os.path.join(self._feat_dir + '_' + self._scaler_type + self._feat_list[0],
                                             self._filenames_list[file_cnt]))
                            if temp_feat.shape[0] < self._feat_cls._max_feat_frames:
                                temp_feat = costume_padding(temp_feat, self._feat_cls._max_feat_frames)

                            if temp_feat.shape[1] <= (self._feat_cls._nfft // 2 + 1):
                                temp_feat = np.concatenate((temp_feat, np.zeros(
                                    (temp_feat.shape[0], (self._feat_cls._nfft // 2 + 1) - temp_feat.shape[1],
                                     temp_feat.shape[-1]))),
                                                           axis=1)
                            else:
                                temp_feat = costume_downsampling(temp_feat, self._feat_cls._nfft // 2 + 1)
                            for feat_num, feat_name in enumerate(self._feat_list):
                                if feat_num == 0:
                                    continue
                                feat = np.load(
                                    os.path.join(self._feat_dir + '_' + self._scaler_type + self._feat_list[0],
                                                 self._filenames_list[file_cnt]))
                                if feat.shape[0] < self._feat_cls._max_feat_frames:
                                    feat = costume_padding(feat, self._feat_cls._max_feat_frames)

                                if feat.shape[1] <= (self._feat_cls._nfft // 2 + 1):
                                    feat = np.concatenate((feat, np.zeros(
                                        (feat.shape[0], (self._feat_cls._nfft // 2 + 1) - feat.shape[1],
                                         feat.shape[-1]))),
                                                          axis=1)
                                    # feat[:, :temp_feat.shape[1], :] = temp_feat
                                else:
                                    feat = costume_downsampling(feat, self._feat_cls._nfft // 2 + 1)

                                temp_feat = np.concatenate((temp_feat, feat), axis=-1)

                        # temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                        temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))

                        for f_row in temp_feat:
                            self._circ_buf_feat.append(f_row)
                        for l_row in temp_label:
                            self._circ_buf_label.append(l_row)

                        # If self._per_file is True, this returns the sequences belonging to a single audio recording
                        if self._per_file:
                            feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                            extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6

                            label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                            extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                            for f_row in extra_feat:
                                self._circ_buf_feat.append(f_row)
                            for l_row in extra_labels:
                                self._circ_buf_label.append(l_row)

                        file_cnt = file_cnt + 1

                    # Read one batch size from the circular buffer
                    feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins, self._nb_ch))
                    label = np.zeros((self._label_batch_seq_len, self._label_len))
                    for j in range(self._feature_batch_seq_len):
                        feat[j, :] = self._circ_buf_feat.popleft()
                    for j in range(self._label_batch_seq_len):
                        label[j, :] = self._circ_buf_label.popleft()
                    # feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins)).transpose(
                    #     (0, 2, 1))

                    # Split to sequences
                    feat = self._split_in_seqs(feat, self._feature_seq_len)
                    # feat = np.transpose(feat, (0, 3, 1, 2))
                    label = self._split_in_seqs(label, self._label_seq_len)
                    if self._is_accdoa:
                        mask = label[:, :, :self._nb_classes]
                        mask = np.tile(mask, 3)
                        label = mask * label[:, :, self._nb_classes:]

                    else:
                        label = [
                            label[:, :, :self._nb_classes],  # SED labels
                            label[:, :, self._nb_classes:] if self._doa_objective is 'mse' else label
                            # SED + DOA labels
                        ]
                    if self._is_online_aug:
                        for jj in range(feat.shape[0]):
                            rand_num = np.random.random()
                            if rand_num > 0.5:
                                aug_rand = np.random.random()
                                if aug_rand > 0.75:
                                    feat[jj, ...] = self.SpecAugmentNp(feat[jj, ...], always_apply=True)
                                elif aug_rand > 0.5:
                                    feat[jj, ...] = self.RandomCutoutNp(feat[jj, ...], always_apply=True)
                                elif aug_rand > 0.25:
                                    feat[jj, ...] = self.RandomShiftUpDownNp(feat[jj, ...], always_apply=True)
                                else:
                                    feat[jj, ...] = self.RandomCutoutHoleNp(feat[jj, ...], always_apply=True)

                    yield feat, label

    def _split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_default_elevation(self):
        return self._default_ele

    def get_azi_ele_list(self):
        return self._feat_cls.get_azi_ele_list()

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._feat_cls.get_hop_len_sec()

    def get_classes(self):
        return self._feat_cls.get_classes()

    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._label_batch_seq_len

    def get_nb_frames(self):
        return self._feat_cls.get_nb_frames()

    def get_data_gen_mode(self):
        return self._is_eval

    def write_output_format_file(self, _out_file, _out_dict):
        return self._feat_cls.write_output_format_file(_out_file, _out_dict)
