# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.
import datetime


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,  # If True: Trains/test on small subset of dataset, and # of epochs

        # INPUT PATH
        dataset_dir='/scratch/asignal/sharath/DCASE2021_SELD_dataset/',
        # Base folder containing the foa_dev/mic_dev and metadata folders

        # OUTPUT PATH
        feat_label_dir='/scratch/asignal/sharath/DCASE2021_SELD_dataset/seld_feat_label/',
        # Directory to dump extracted features and labels
        model_dir='models/',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='results/',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        # DNN MODEL PARAMETERS
        is_accdoa=True,  # True: Use ACCDOA output format
        doa_objective='mse',
        # if is_accdoa=True this is ignored, otherwise it supports: mse, masked_mse. where mse- original seld approach; masked_mse - dcase 2020 approach

        label_sequence_length=60,  # Feature sequence length
        batch_size=256,  # Batch size
        dropout_rate=0.05,  # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,  # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],
        # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        rnn_size=[128, 128],  # RNN contents, length of list = number of layers, list value = number of nodes
        fnn_size=[128],  # FNN contents, length of list = number of layers, list value = number of nodes
        loss_weights=[1., 1000.],  # [sed, doa] weight for scaling the DNN outputs
        nb_epochs=50,  # Train for maximum epochs
        epochs_per_fit=5,  # Number of epochs per fit

        # METRIC PARAMETERS
        lad_doa_thresh=20,

        # ADDED PARAMETERS BY MASOUD MOHTADIFAR
        scaler_type=['standard', 'minmax'],  # have to be one of these 2 or the code will not give desired results
        wavelet_family='db10',
        cwavelet_family='morl',
        wavelet_mode='constant',
        is_mel=True,
        scatter_wavelet_Q=126,
        scatter_wavelet_J=9,
        nv_num=8,
        is_online_aug=False,
        is_acs_aug=False
    )
    Q = params['scatter_wavelet_Q']
    params['feature_list'] = ['_spec', '_IPD', '_IPD_Cos', '_IPD_Sin', '_IV', f'_scatter_wavelet_Q{Q}_order_1',
                              f'_scatter_wavelet_Q{Q}_order_2', f'_scatter_wavelet_Q{Q}', '_CWT_abs', '_SSQ_abs',
                              '_CWT_IPD', '_SSQ_IPD', '_CWT_IPD_Cos', '_CWT_IPD_Sin', '_SSQ_IPD_Cos', '_SSQ_IPD_Sin']
    # params['feature_list'] = ['_CWT_abs', '_SSQ_abs',
    #                           '_CWT_IPD', '_SSQ_IPD', '_CWT_IPD_Cos', '_CWT_IPD_Sin', '_SSQ_IPD_Cos', '_SSQ_IPD_Sin']
    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]  # CNN time pooling
    params['patience'] = int(params['nb_epochs'])  # Stop training if patience is reached

    params['unique_classes'] = {
        'alarm': 0,
        'baby': 1,
        'crash': 2,
        'dog': 3,
        'female_scream': 4,
        'female_speech': 5,
        'footsteps': 6,
        'knock': 7,
        'male_scream': 8,
        'male_speech': 9,
        'phone': 10,
        'piano': 11  # , 'engine':12, 'fire':13
    }

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        params['mode'] = 'dev'
        params['dataset'] = 'mic'

    elif argv == '3':
        params['mode'] = 'eval'
        params['dataset'] = 'mic'

    elif argv == '4':
        params['mode'] = 'dev'
        params['dataset'] = 'foa'

    elif argv == '5':
        params['mode'] = 'eval'
        params['dataset'] = 'foa'

    elif argv == '11':
        params['quick_test'] = False  # If True: Trains/test on small subset of dataset, and # of epochs

        # INPUT PATH
        params['dataset_dir'] = 'E:\Datasets\SELD2021\DATASET'
        # Base folder containing the foa_dev/mic_dev and metadata folders

        # OUTPUT PATH
        params['feat_label_dir'] = 'E:\Datasets\SELD2021\seld_feat_label_quick_temp'
        params['nv_num'] = 8
        # Directory to dump extracted features and labels

        # DATASET LOADING PARAMETERS
        params['mode'] = 'eval'  # 'dev' - development or 'eval' - evaluation dataset

        params['scaler_type'] = ['standard']  # have to be one of these 2 or the code will not give desired results
        params['is_online_aug'] = True
        is_acs_aug = False
        params['feature_list'] = ['_spec', '_IPD']

    elif argv == '12':
        params['quick_test'] = False  # If True: Trains/test on small subset of dataset, and # of epochs

        # INPUT PATH
        params['dataset_dir'] = '/home/ubuntu/DCASE-2021/DATASET'
        # Base folder containing the foa_dev/mic_dev and metadata folders

        # OUTPUT PATH
        params['feat_label_dir'] = '/home/ubuntu/DCASE-2021/features/seld_feat_label'
        # Directory to dump extracted features and labels

        # DATASET LOADING PARAMETERS
        params['mode'] = 'eval'  # 'dev' - development or 'eval' - evaluation dataset
        params['nv_num'] = 32
        params['model_dir'] = "/home/ubuntu/DCASE-2021/models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        params['dcase_output_dir'] = "/home/ubuntu/DCASE-2021/results/" + datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S")

    elif argv == '13':
        params['feature_list'] = ['_IPD', '_IV']
        params['quick_test'] = True  # If True: Trains/test on small subset of dataset, and # of epochs

        # INPUT PATH
        params['dataset_dir'] = 'E:\Datasets\SELD2021\DATASET'
        # Base folder containing the foa_dev/mic_dev and metadata folders

        # OUTPUT PATH
        params['feat_label_dir'] = 'E:\Datasets\SELD2021\seld_feat_label_quick_temp_ipd'
        params['nv_num'] = 8
        # Directory to dump extracted features and labels

        # DATASET LOADING PARAMETERS
        params['mode'] = 'eval'  # 'dev' - development or 'eval' - evaluation dataset

        params['scaler_type'] = ['standard']  # have to be one of these 2 or the code will not give desired results
        params['is_online_aug'] = True
        is_acs_aug = False

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True
        params['epochs_per_fit'] = 1

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
