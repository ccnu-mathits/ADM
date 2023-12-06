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

class SymbolicLawParser:
    def __init__(self,
                 data_name = "",
                 data_heads = ["sum_g","sum_s1","sum_s2","count","interval"],
                 select_feature_num = 2,
                 x1_weight = [],
                 x2_weight = [],
                 p_niterations = 20000,
                 p_populations = 50,
                 p_binary_operators = ["+","-","*","pow"],
                 p_unary_operators = ["cos","sin","log","exp"],
                 p_nested = True,
                 p_model_selection = "accuracy",
                 p_loss = "loss(x, y) = (x - y)^2",
                 p_batching = False,
                 p_batch_size = 128,
                 p_maxsize = 15,
                 ):
        self.data_name = data_name
        self.data_heads = data_heads
        self.select_feature_num = select_feature_num
        self.x1_weight = x1_weight
        self.x2_weight = x2_weight
        self.p_niterations = p_niterations
        self.p_populations = p_populations
        self.p_binary_operators = p_binary_operators
        self.p_unary_operators = p_unary_operators
        self.p_model_selection = p_model_selection
        self.p_loss = p_loss
        self.p_batching = p_batching
        self.p_batch_size = p_batch_size
        self.p_maxsize = p_maxsize

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

        print("Dataset:", self.data_name)
        encodings_1 = pandas.read_csv('./StateMLPData/' + data_name +"_b_encoding1.csv",
                               header=None, names=data_heads)
        encodings_2 = pandas.read_csv('./StateMLPData/' + data_name + "_b_encoding2.csv",
                                      header=None, names=data_heads)
        states_1 = pandas.read_csv('./StateMLPData/' + data_name + "_states1.csv",
                                      header=None, names=['state1'])
        states_2 = pandas.read_csv('./StateMLPData/' + data_name + "_states2.csv",
                                      header=None, names=['state2'])

        x1_index = heapq.nlargest(select_feature_num, range(len(x1_weight)), x1_weight.__getitem__)
        x2_index = heapq.nlargest(select_feature_num, range(len(x2_weight)), x2_weight.__getitem__)
        x1_select_feature_names = [str(data_heads[i]) for i in x1_index]
        x2_select_feature_names = [str(data_heads[i]) for i in x2_index]

        print("Selected Feature for SKILL 1:",x1_select_feature_names)
        print("Selected Feature for SKILL 2:",x2_select_feature_names)

        self.y1 = states_1["state1"]
        self.y2 = states_2["state2"]
        self.X1 = encodings_1[x1_select_feature_names]
        self.X2 = encodings_2[x2_select_feature_names]

    def parse(self):
        self.model1 = PySRRegressor(
            equation_file="./PYSRres/STATE1-" + self.data_name+".csv",
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
            # procs = 15,
            # maxdepth = 3,
        )
        print("Symbolic Regression for the learning rule of SKILL-1 !!!")
        self.model1.fit(self.X1, self.y1)

        self.model2 = PySRRegressor(
            equation_file="./PYSRres/STATE2-" + self.data_name+".csv",
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
            # procs = 15,
            # maxdepth = 3,
        )
        print("Symbolic Regression for the learning rule of SKILL-2 !!!")
        self.model2.fit(self.X2, self.y2)

        print("Input Features:",self.data_heads)
        print("Feature Weights for SKILL 1:",self.x1_weight)
        print("Feature Weights for SKILL 2:",self.x2_weight)
        print("-" * 30 + "Symbolic Model for SKILL-1" + "-" * 30)
        print(self.model1.equations)
        print("-" * 30 + "--------------------------" + "-" * 30)
        print("-" * 30 + "Symbolic Model for SKILL-2" + "-" * 30)
        print(self.model2.equations)
        print("-" * 30 + "--------------------------" + "-" * 30)


if __name__ == '__main__':
    #Parametergrqoups [alpha,beta,gamma]
    parms = {}
    parms["Linear"] = [1,0.3,0.1]
    parms["Exponential"] = [0.1,1,0.1]
    parms["Power"] = [0.5,2,0.1]

    SKILL1 = "Power" #[Linear, Power, Exponential]
    SKILL2 = "Linear" #[Linear, Power, Exponential]

    tr = TransformerRegressor(
        g_theta=0,
        g_m_factor=0.5,
        g_n_exercise=20,
        g_n_student=50,
        g_n_length=20,
        m_att_hidden_num=100,
        m_state_hidden_num=1000,
        m_att_reg=1,
        early_stop=500,
        learning_rate=0.0005,
        device="cuda:0",
        g_learingrule1=SKILL1,
        g_alpha1=parms[SKILL1][0],
        g_beta1=parms[SKILL1][1],
        g_gamma1=parms[SKILL1][2],
        g_learingrule2=SKILL2,
        g_alpha2=parms[SKILL2][0],
        g_beta2=parms[SKILL2][1],
        g_gamma2=parms[SKILL2][2],
    )
    tr.train()
    tr.load_bast_model()
    feature_importance_state1, feature_importance_state2 = tr.save_data_in_state_MLP()

    slp = SymbolicLawParser(
        data_name=tr.model_path,
        x1_weight=feature_importance_state1,
        x2_weight=feature_importance_state2,
        p_binary_operators=["+", "-", "*", "pow"],
        p_unary_operators=["exp"],
        p_niterations=2000,
        p_populations=50,
        p_nested=True,
        p_model_selection="best",
        p_loss="loss(x, y) = abs(x - y)",
        p_batching=False,
        p_batch_size=512,
        p_maxsize=15,
    )
    slp.parse()
