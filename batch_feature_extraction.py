# This part is for installing the required packages further in the code
import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


for package in ['librosa', 'kymatio', 'ssqueezepy', 'cupy-cuda11x']:
    install(package=package)

# Extracts the features, labels, and normalizes the development and evaluation split features.

import cls_feature_class
import parameter

process_str = 'dev, eval'  # 'dev' or 'eval' will extract features for the respective set accordingly
#  'dev, eval' will extract features of both sets together

params = parameter.get_params('13')

if 'dev' in process_str:
    # -------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(params, is_eval=False)

    # Extract features and normalize them
    # dev_feat_cls.extract_all_feature()
    # dev_feat_cls.preprocess_features()
    dev_feat_cls.extract_spectrogram_etc()
    dev_feat_cls.preprocess_spec_etc()
    # dev_feat_cls.extract_scatter_wavelet()
    # dev_feat_cls.preprocess_scatter_wavelet()
    # dev_feat_cls.extract_cwt_ssq(nv_num=params['nv_num'])
    # dev_feat_cls.preprocess_cwt_ssq()
    # dev_feat_cls.extract_norm_mel_for_spec_etc()
    # dev_feat_cls.extract_mel_for_cwt(nv_num=params['nv_num'])
    # dev_feat_cls.extract_mel_for_scatter()
    # save the features shape as images
    # dev_feat_cls.save_feature_plot(1)

    # # Extract labels in regression mode
    # dev_feat_cls.extract_all_labels()

if 'eval' in process_str:
#     # -----------------------------Extract ONLY features for evaluation set-----------------------------
    eval_feat_cls = cls_feature_class.FeatureClass(params, is_eval=True)
#
#     # Extract features and normalize them
#     eval_feat_cls.extract_all_feature()
#     eval_feat_cls.preprocess_features()
#     eval_feat_cls.extract_spectrogram_etc()
#     eval_feat_cls.preprocess_spec_etc()
#     eval_feat_cls.extract_scatter_wavelet()
#     eval_feat_cls.preprocess_scatter_wavelet()
#     eval_feat_cls.extract_cwt_ssq(nv_num=params['nv_num'])
#     eval_feat_cls.preprocess_cwt_ssq()
#     eval_feat_cls.extract_norm_mel_for_spec_etc()
#     eval_feat_cls.extract_mel_for_cwt(nv_num=params['nv_num'])
#     eval_feat_cls.extract_mel_for_scatter()
