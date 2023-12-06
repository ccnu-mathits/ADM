# -*- coding: utf-8 -*-

"""

--------------------------------------

    Author: Xiaoxuan Shen
    
    Date:   2022/6/6 10:57
    
--------------------------------------

"""
import torch
import torch.optim as opt

class Trasnsformer_model(torch.nn.Module):
    def __init__(self,D_UF,D_GF,D_CF,D_SKILL,UF,GF,GS,SKILLS,
                 ATT_HIDDEN_SIZE,SQUENCE_LENGTH,STATE_HIDDEN_SIZE,
                 ATT_DROPRATE,STATE_DROPRATE,DEVICE):
        super(Trasnsformer_model, self).__init__()
        self.UF = UF.to(DEVICE)
        self.GF = GF.to(DEVICE)
        self.GS = GS.to(DEVICE)
        self.SKILLS = SKILLS
        self.user_num = UF.shape[0]
        self.game_num = GF.shape[0]

        self.encoders = []
        self.transformers = []
        self.encoding_norms = []
        for s in self.SKILLS:
            self.encoders.append(torch.nn.Sequential(
                torch.nn.BatchNorm1d(D_UF + D_GF + D_CF),
                torch.nn.Linear(D_UF+D_GF+D_CF,ATT_HIDDEN_SIZE),
                torch.nn.LeakyReLU(),
                # torch.nn.Tanh(),
                torch.nn.Dropout(ATT_DROPRATE),
                torch.nn.Linear(ATT_HIDDEN_SIZE,1),
            ).to(DEVICE))
            self.encoding_norms.append(torch.nn.BatchNorm1d(D_UF+D_GF+D_CF,affine=False).to(DEVICE))
            self.transformers.append(torch.nn.Sequential(
                torch.nn.Linear(D_UF+D_GF+D_CF, STATE_HIDDEN_SIZE),
                torch.nn.LeakyReLU(),
                # torch.nn.Tanh(),
                torch.nn.Dropout(STATE_DROPRATE),
                torch.nn.Linear(STATE_HIDDEN_SIZE, STATE_HIDDEN_SIZE),
                torch.nn.Linear(STATE_HIDDEN_SIZE, STATE_HIDDEN_SIZE),
                torch.nn.LeakyReLU(),
                # torch.nn.Tanh(),
                torch.nn.Dropout(STATE_DROPRATE),
                torch.nn.Linear(STATE_HIDDEN_SIZE, 1),
            ).to(DEVICE))

        self.user_init = torch.nn.Parameter(torch.ones(self.user_num, D_SKILL)*0)
        self.register_parameter('user_init', self.user_init)
        # difficulty parameters in IRT
        self.CD_diff = torch.nn.Parameter(torch.zeros(self.game_num,D_SKILL))
        self.register_parameter('CD_diff', self.CD_diff)
        # discrimination parameters in IRT
        self.CD_disc = torch.nn.Parameter(torch.ones(self.game_num, 1)*0.2)
        self.register_parameter('CD_disc', self.CD_disc)
        # guess parameters in IRT
        self.CD_guess = torch.nn.Parameter(torch.zeros(self.game_num, 1))
        self.register_parameter('CD_guess', self.CD_guess)


    def forward(self, user,questions,Context):
        length = questions.shape[0]
        # Games skills [length, D_SKILL]
        Game_skill = self.GS[questions,:]
        # Games feature [length, D_GF]
        Game_feature = self.GF[questions,:]
        # Game difficulty [length, D_SKILL]
        Game_diff = self.CD_diff[questions,:]
        # Game discrimination [length, 1]
        Game_disc = self.CD_disc[questions, :]
        # Game guess [length, 1]
        CD_guess = self.CD_guess[questions, :]
        # User feature [1, D_UF]
        User_feature = self.UF[user,:].repeat(length,1)
        # User init cognitive state
        User_init_cognitive_state = self.user_init[user,:]
        # Context feature [length, D_CF]

        # Context = Context / length * 10
        encoder_inputs = torch.cat((User_feature, Game_feature,Context), 1)
        att_all_list = []
        transformer_inputs = []
        transformer_inputs_n = []
        cognitive_states = []
        for skill_i in range(self.SKILLS.__len__()):
            # compute the attention value of learning behaviors
            encoder_i = self.encoders[skill_i]
            att_i = encoder_i.forward(encoder_inputs)
            att_all_list.append(att_i)
            # encode the Game_feature and Context_feature
            att_matrix_i = torch.triu(att_i.repeat(1,length))
            encoded_game_feature = torch.mm(Game_feature.t(),att_matrix_i).t()
            # transform the encoded behavior to cognitive state
            transformers_i = self.transformers[skill_i]
            transformer_input = torch.cat(
                (User_feature,encoded_game_feature,Context),1)
            # norm the encodings
            normer = self.encoding_norms[skill_i]
            transformer_input_n = normer(transformer_input)
            transformer_inputs.append(transformer_input)
            transformer_inputs_n.append(transformer_input_n)
            cognitive_state_i = transformers_i.forward(transformer_input_n)
            cognitive_states.append(cognitive_state_i)


        # the increasement of cognitive_state
        cognitive_state_b = torch.cat(cognitive_states,1)
        # the attention value of learning behaviors for all skills
        att_all = torch.cat(att_all_list, 1)
        # cognitive_state
        cognitive_state = cognitive_state_b + User_init_cognitive_state
        cognitive_state = torch.cat([User_init_cognitive_state.unsqueeze(0),cognitive_state],0)

        # IRT model (3 parameters)
        cognitive_state_length = cognitive_state[:-1,:]
        p_correctness = CD_guess + (1-CD_guess)/\
                        (1+torch.exp(-1.702*Game_disc*(((cognitive_state_length - Game_diff)
                                                                      * Game_skill).sum(1).unsqueeze(1))))

        # IRT model (2 parameters)
        # p_correctness = 1/(1 + torch.exp(-1.702 * Game_disc * (((cognitive_state_length - Game_diff)
        #                                                       * Game_skill).sum(1).unsqueeze(1))))

        # Rasch model
        # cog = ((cognitive_state_length - Game_diff) * Game_skill).sum(1).unsqueeze(1)
        # p_correctness = torch.exp(cog) / 1 + torch.exp(cog)
        return encoder_inputs, att_all_list,transformer_inputs,transformer_inputs_n,cognitive_states, att_all, p_correctness