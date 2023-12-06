# -*- coding: utf-8 -*-

"""

--------------------------------------

    Author: Xiaoxuan Shen
    
    Date:   2022/6/6 10:57
    
--------------------------------------

"""
import torch


class Trasnsformer_model(torch.nn.Module):
    def __init__(self,D_Q,D_C,D_S,ATT_HIDDEN_SIZE,MAX_LENGTH,STATE_HIDDEN_SIZE,DEVICE):
        super(Trasnsformer_model, self).__init__()

        self.user_init_representation = torch.nn.Parameter(torch.zeros(1,D_S))
        self.ATT_HIDDEN_SIZE = ATT_HIDDEN_SIZE
        self.STATE_HIDDEN_SIZE = STATE_HIDDEN_SIZE
        self.D_S = D_S

        self.att1_MLP = torch.nn.Sequential(
            torch.nn.BatchNorm1d(D_Q+D_C),
            torch.nn.Linear(D_Q+D_C,ATT_HIDDEN_SIZE),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(ATT_HIDDEN_SIZE,1),
        )
        self.att2_MLP = torch.nn.Sequential(
            torch.nn.BatchNorm1d(D_Q + D_C),
            torch.nn.Linear(D_Q + D_C, ATT_HIDDEN_SIZE),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(ATT_HIDDEN_SIZE, 1),
        )

        self.encoding1_nomer = torch.nn.BatchNorm1d(D_Q + D_C,affine=False).to(DEVICE)
        self.encoding2_nomer = torch.nn.BatchNorm1d(D_Q + D_C,affine=False).to(DEVICE)

        self.state1_MLP = torch.nn.Sequential(
            torch.nn.Linear(D_Q + D_C, STATE_HIDDEN_SIZE),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(STATE_HIDDEN_SIZE, STATE_HIDDEN_SIZE),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(STATE_HIDDEN_SIZE, 1),
        )
        self.state2_MLP = torch.nn.Sequential(
            torch.nn.Linear(D_Q + D_C, STATE_HIDDEN_SIZE),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(STATE_HIDDEN_SIZE, STATE_HIDDEN_SIZE),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(STATE_HIDDEN_SIZE, 1),
        )

        self.CD_guess = torch.nn.Parameter(torch.ones(1)*0.1)
        self.CD_d = -1.702


    def forward(self, Questions,Contexts, Q_info):
        # Questions: [n_length, D_Q]
        # Contexts: [n_length, D_C]

        length = Questions.size(0)

        # I: [n_length, D_Q + D_C]
        I = torch.cat([Questions,Contexts],1)

        att_1 = self.att1_MLP.forward(I)
        att_matrix_1 = torch.triu(att_1.repeat(1, length))
        encoded_game_feature_1 = torch.mm(Questions.t(), att_matrix_1).t()
        transformer_input_1 = torch.cat((encoded_game_feature_1, Contexts), 1)
        # transformer_input_1n = self.encoding1_nomer(transformer_input_1)
        transformer_input_1n = transformer_input_1
        cognitive_state_1 = self.state1_MLP.forward(transformer_input_1n)

        att_2 = self.att2_MLP.forward(I)
        att_matrix_2 = torch.triu(att_2.repeat(1, length))
        encoded_game_feature_2 = torch.mm(Questions.t(), att_matrix_2).t()
        transformer_input_2 = torch.cat((encoded_game_feature_2, Contexts), 1)
        # transformer_input_2n = self.encoding2_nomer(transformer_input_2)
        transformer_input_2n = transformer_input_2
        cognitive_state_2 = self.state2_MLP.forward(transformer_input_2n)

        att_all_list = [att_1, att_2]
        transformer_inputs = [transformer_input_1,transformer_input_2]
        transformer_inputs_n = [transformer_input_1n,transformer_input_2n]
        cognitive_states = [cognitive_state_1,cognitive_state_2]

        # the increasement of cognitive_state
        cognitive_state_b = torch.cat(cognitive_states, 1)
        # the attention value of learning behaviors for all skills
        att_all = torch.cat(att_all_list, 1)
        # cognitive_state
        cognitive_state = cognitive_state_b + self.user_init_representation
        cognitive_state = torch.cat([self.user_init_representation, cognitive_state], 0)

        # u_state: n_length*D_S
        u_state = cognitive_state[:-1,:]
        n_skill = u_state.size(1)
        q_diff = Q_info[:,0:n_skill]
        q_Q = Q_info[:,n_skill:2*n_skill]
        # IRT model (3 parameters)
        p_correctness = self.CD_guess + (1-self.CD_guess)/(1+torch.exp(self.CD_d*(((u_state - q_diff)*q_Q).sum(1).unsqueeze(1))))

        return I, att_all_list,transformer_inputs,transformer_inputs_n,cognitive_states, att_all, p_correctness