# -*- coding: utf-8 -*-

"""

--------------------------------------

    Author: Xiaoxuan Shen
    
    Date:   2023/2/8 10:00

--------------------------------------

"""

import numpy as np
import pandas
import warnings
import heapq
from pysr import PySRRegressor
import pysr
from DeepRegressor import TransformerRegressor
import os
import random

class SymbolicLawParser:
    def __init__(self,
                 data_name = "",
                 dataset = {},
                 select_feature_num = 3,
                 feature_importances = {},
                 sampled_data_num = 100,
                 p_niterations = 20000,
                 p_populations = 50,
                 p_binary_operators = ["+","-","*","pow"],
                 p_unary_operators = ["cos","sin","log","exp"],
                 p_nested = True,
                 p_model_selection = "best",
                 # p_loss = "loss(x, y) = (x - y)^2",
                 p_loss = "loss(x, y) = abs(x - y)",
                 p_batching = False,
                 p_batch_size = 128,
                 p_maxsize = 15,
                 p_procs=20,
                 ):
        self.data_name = data_name
        self.dataset = dataset
        self.select_feature_num = select_feature_num
        self.sampled_data_num = sampled_data_num
        self.feature_importances = feature_importances
        self.p_niterations = p_niterations
        self.p_populations = p_populations
        self.p_binary_operators = p_binary_operators
        self.p_unary_operators = p_unary_operators
        self.p_model_selection = p_model_selection
        self.p_loss = p_loss
        self.p_batching = p_batching
        self.p_batch_size = p_batch_size
        self.p_maxsize = p_maxsize
        self.p_procs = p_procs

        self.p_nested_constraints = {}
        if p_nested:
            operators = p_binary_operators + p_unary_operators
            operators.remove("+")
            operators.remove("-")
            operators.remove("*")
            constraints = {}
            for op in operators:
                constraints[op] = 0
            for op in operators:
                self.p_nested_constraints[op] = constraints

    def parse(self):
        print('Start parsing the symbolic laws for',self.data_name)
        skills = list(self.dataset.keys())
        print('SKILLS:',skills)
        for s in skills:
            if not os.path.exists("./PYSRres/"+s):
                os.makedirs("./PYSRres/"+s)
            srmodel = PySRRegressor(
                equation_file="./PYSRres/"+s+'/' + self.data_name+"_f"+str(self.select_feature_num)+".csv",
                niterations=self.p_niterations,
                populations=self.p_populations,
                binary_operators=self.p_binary_operators,
                unary_operators=self.p_unary_operators,
                nested_constraints=self.p_nested_constraints,
                model_selection=self.p_model_selection,
                loss=self.p_loss,  # Custom loss function (julia syntax)
                update=False,
                batching=self.p_batching,
                batch_size=self.p_batch_size,
                maxsize=self.p_maxsize,
                # complexity_of_constants=1.2,
                procs = self.p_procs,
                # maxdepth = 3,
            )

            print("-" * 30 + "--------------------------" + "-" * 30)
            print("Symbolic Regression for the learning rule of",s,"!!!")
            print("-" * 30 + "--------------------------" + "-" * 30)

            # Choose features
            f_i_pd = self.feature_importances[s]
            f_i_head = list(f_i_pd.axes[1])
            f_i_value = f_i_pd.values.tolist()[0]
            f_i_index = heapq.nlargest(self.select_feature_num, range(len(f_i_value)), f_i_value.__getitem__)
            selected_features = [str(f_i_head[i]) for i in f_i_index]
            print("Selected Features for SKILL",s,":")
            for i in selected_features:
                print(i)
            X = self.dataset[s]['encodings']
            Y = self.dataset[s]['states']
            X_selected = X[selected_features]

            sample_index = random.sample(range(X.shape[0]), self.sampled_data_num)
            sX_selected = X_selected.iloc[sample_index]
            sY = Y.iloc[sample_index]

            srmodel.fit(sX_selected, sY)





