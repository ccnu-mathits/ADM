# -*- coding: utf-8 -*-

"""

--------------------------------------

    Author: Xiaoxuan Shen
    
    Date:   2023/3/24 11:37
    
--------------------------------------

"""
import pysr
from LawParser import SymbolicLawParser
from DeepRegressor import TransformerRegressor
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    tr = TransformerRegressor(
        d_sequence_length=2000,
        m_att_hidden_num=1000,
        m_state_hidden_num=1000,
        m_encoder_reg=1,
        max_epoch=50,
        early_stop=10,
        learning_rate=1e-4,
        sampled_usernum=1000,
        device='cuda:0',
    )
    ############################################################################
    # Training the model
    ############################################################################
    tr.train()

    ############################################################################
    # generating the encoded data and feature importance, then save them to .csv
    ############################################################################
    tr.load_best_model()
    encodings, feature_importances = tr.save_encoded_and_state_data(vis=False)

    ############################################################################
    # If the encoded data is generated and saved, you can load the data from .csv
    ############################################################################
    # encodings, feature_importances = tr.load_encoded_and_state_data()
    #
    slp = SymbolicLawParser(
        data_name=tr.model_name,
        dataset=encodings,
        feature_importances=feature_importances,
        select_feature_num=8,
        sampled_data_num=2000,
        p_binary_operators=["+", "-", "*", "pow"],
        p_unary_operators=["exp","log","sin"],
        p_niterations=2000,
        p_populations=100,
        p_nested=True,
        p_loss="loss(x, y) = abs(x - y)",
        p_batching=False,
        p_maxsize=15,
        p_procs = 24,
    )
    slp.parse()
