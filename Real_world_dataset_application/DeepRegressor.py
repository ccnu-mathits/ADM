# -*- coding: utf-8 -*-

"""

--------------------------------------

    Author: Xiaoxuan Shen
    
    Date:   2023/2/7 14:53
    
--------------------------------------

"""
import torch
from Utils import load_dataset
from TorchModels import Trasnsformer_model
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import torch.optim as opt
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class TransformerRegressor:
    def __init__(self,
                 d_path = '../Dataset/Lumosity/processed_data',
                 d_sequence_length = 300,
                 d_score_min = 0.01,
                 d_score_max = 0.8,
                 m_att_hidden_num = 50,
                 m_att_droprate = 0,
                 m_state_hidden_num = 50,
                 m_state_droprate=0,
                 m_encoder_reg = 1,
                 max_epoch=10000,
                 early_stop=50,
                 learning_rate=0.01,
                 sampled_usernum = 100,
                 device='cpu',
                 ):
        self.data_path = d_path
        self.data_sq_length = d_sequence_length
        self.m_att_hidden_num = m_att_hidden_num
        self.m_att_droprate = m_att_droprate
        self.m_state_hidden_num = m_state_hidden_num
        self.m_state_droprate = m_state_droprate
        self.m_att_reg = m_encoder_reg
        self.max_epoch = max_epoch
        self.early_stop = early_stop
        self.lr = learning_rate
        self.sampled_usernum = sampled_usernum
        self.device = device

        self.model_name = 'Lumosity'+str(d_sequence_length)+'_max'+str(d_score_max)+'min'+str(d_score_min)+'_net'+str(m_att_hidden_num)+'-'+str(m_state_hidden_num)+'_att_reg'+str(m_encoder_reg)
        self.writer = SummaryWriter(log_dir=('runs/'+self.model_name))

        users, games, squences = load_dataset(d_path,d_sequence_length,d_score_min,d_score_max)
        self.user_feature = torch.tensor(users['user_info'])
        self.user_feature_head = users['user_heads']
        self.user_num = users['user_num']
        self.game_feature = torch.tensor(games['game_features'])
        self.game_feature_head = games['game_heads']
        self.skill_head = [self.game_feature_head[i] for i in range(7)]
        self.skill_game = self.game_feature[:, 0:7]
        # Add the G skill
        self.game_feature_head.insert(0,'General')
        self.game_feature = \
            torch.cat((torch.ones(self.game_feature.shape[0],1),self.game_feature),1)
        self.game_num = games['game_num']
        self.squences = {}
        self.contexts = {}
        self.contexts_heads = ['game_count', 'area_count', 'attribute_count',
                               'game_interval', 'area_interval', 'attribute_interval']
        for i in squences:
            self.squences[int(i)] = {}
            self.squences[int(i)]['sq_game'] = torch.tensor(squences[i]['q_squences']).to(self.device)
            self.squences[int(i)]['sq_level'] = torch.tensor(squences[i]['level_squences']).to(self.device)
            self.squences[int(i)]['sq_score'] = (torch.tensor(squences[i]['score_squences']).unsqueeze(1)).to(self.device)
            contexts_i = [squences[i][h] for h in self.contexts_heads]
            self.contexts[int(i)] = torch.tensor(contexts_i).t().to(self.device)

        self.dl_model = Trasnsformer_model(
            D_UF=self.user_feature_head.__len__(),
            D_GF=self.game_feature_head.__len__(),
            D_CF=self.contexts_heads.__len__(),
            D_SKILL=self.skill_head.__len__(),
            UF=self.user_feature,
            GF=self.game_feature,
            GS=self.skill_game,
            SKILLS=self.skill_head,
            ATT_HIDDEN_SIZE=self.m_att_hidden_num,
            SQUENCE_LENGTH=self.data_sq_length,
            STATE_HIDDEN_SIZE=self.m_state_hidden_num,
            ATT_DROPRATE=self.m_att_droprate,
            STATE_DROPRATE=self.m_state_droprate,
            DEVICE=self.device
        ).to(self.device)
        # gather all the parameters
        self.params = [{'params': self.dl_model.parameters(),'lr':self.lr*10}]
        for encoder in self.dl_model.encoders:
            self.params.append({'params': encoder.parameters()})
        for transformer in self.dl_model.transformers:
            self.params.append({'params': transformer.parameters()})
        for normer in self.dl_model.encoding_norms:
            self.params.append({'params': normer.parameters()})


    def train(self):
        # Chose optimizer and training scheduler
        # optimizer = opt.Adam(self.params, lr=self.lr)
        optimizer = opt.RMSprop(self.params, lr=self.lr)
        # optimizer = opt.SGD(self.params, lr=self.lr)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

        best_loss = 10
        stop = 0


        for e in range(self.max_epoch):
            self.dl_model.train()
            fit_loss = 0
            att_reg = 0
            for user in tqdm(list(self.squences.keys())):
            # for user in self.squences:
                questions = self.squences[user]['sq_game']
                scores = self.squences[user]['sq_score']
                context = self.contexts[user]
                _, _, _, _, _, att_all, p_correctness\
                    =self.dl_model.forward(user,questions,context)
                score_fit_loss = (scores - p_correctness).abs().mean()
                if torch.isnan(score_fit_loss):
                    quit()
                fit_loss += score_fit_loss.detach()
                att_smooth_loss = (att_all-1).square().mean()
                att_reg += att_smooth_loss.detach()
                total_loss = score_fit_loss + self.m_att_reg*att_smooth_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            train_fit_loss = (fit_loss / self.squences.__len__()).cpu().numpy()
            train_att_reg = (att_reg / self.squences.__len__()).cpu().numpy()
            train_total_loss = train_fit_loss + self.m_att_reg*train_att_reg
            # scheduler.step()
            print('epoch:', e, '| Total_loss:', train_total_loss, '| Fit_loss:', train_fit_loss, '| Att_reg:', train_att_reg)
            self.writer.add_scalar('Fit_loss',train_fit_loss, e)
            self.writer.add_scalar('Att_reg', train_att_reg, e)

            if train_total_loss < best_loss:
                best_loss = train_total_loss
                stop = 0
                self.save_model()
            else:
                stop += 1
                if stop == self.early_stop:
                    break

    def save_model(self):
        if not os.path.exists("./SavedModels"):
            os.makedirs("./SavedModels")
        torch.save(self.dl_model, "./SavedModels/"+self.model_name+".pth")
        print("Model is saved in:")
        print("./SavedModels/"+self.model_name+".pth")

    def load_best_model(self):
        self.best_model = torch.load("./SavedModels/"+self.model_name+".pth")
        print("The best model is loaded from:")
        print("./SavedModels/"+self.model_name+".pth")

    def save_errors(self):
        skill_list = self.skill_game.nonzero()[:,1]
        test_q = []
        test_e = []
        self.best_model.eval()
        user_list = list(self.squences.keys())
        if self.sampled_usernum>0:
            self.sampled_user_list = random.sample(user_list,self.sampled_usernum)
        else:
            self.sampled_user_list = user_list
        for user in tqdm(self.sampled_user_list):
            questions = self.squences[user]['sq_game']
            scores = self.squences[user]['sq_score']
            context = self.contexts[user]
            _, _, _, _, _, att_all, p_correctness \
                = self.best_model.forward(user, questions, context)
            error = (scores - p_correctness).abs().detach()
            test_q.append(questions.unsqueeze(1))
            test_e.append(error)
        test_q = torch.cat(test_q, 0).cpu().numpy()
        test_e = torch.cat(test_e, 0).cpu().numpy()
        test_s = skill_list[test_q].numpy()
        df = pd.DataFrame({'game_id':np.squeeze(test_q),
                           'abs_error':np.squeeze(test_e),
                           'area':np.squeeze(test_s)})
        df.to_csv('error.csv',index=False,sep=',')
        print(df)






    def save_encoded_and_state_data(self,vis=False):
        print('Computing the encodings, states and feature importances...')
        self.best_model.eval()
        model_device = self.best_model.CD_diff.device
        encodings = {}
        for i in range(self.skill_head.__len__()):
            skill = self.skill_head[i]
            encodings[skill] = {}
            encodings[skill]['encodings'] = []
            encodings[skill]['states'] = []

        # Sampled some users for symbolic regression
        user_list = list(self.squences.keys())
        if self.sampled_usernum>0:
            self.sampled_user_list = random.sample(user_list,self.sampled_usernum)
        else:
            self.sampled_user_list = user_list
        self.encoded_behavior_heads = self.user_feature_head+\
                                      ['sum_'+i for i in self.game_feature_head]+\
                                      self.contexts_heads

        feature_importances = {}
        for user in tqdm(self.sampled_user_list):
            questions = self.squences[user]['sq_game'].to(model_device)
            context = self.contexts[user].to(model_device)
            _, _, b_encodings, b_encodings_n, p_states, _, _ \
                = self.best_model.forward(user, questions,context)
            for i in range(self.skill_head.__len__()):
                skill = self.skill_head[i]
                encodings[skill]['encodings'].append(b_encodings[i].detach().cpu().numpy())
                state_i = p_states[i].detach().cpu().numpy()
                encodings[skill]['states'].append(state_i)

                if vis:
                    if not os.path.exists("./StateData/" + skill+"/"+self.model_name+"/vis"):
                        os.makedirs("./StateData/" + skill+"/"+self.model_name+"/vis")
                    fig = plt.figure()
                    plt.plot(state_i)
                    fig.savefig("./StateData/" + skill+"/"+self.model_name+"/vis/"+str(user)+'.png')

                encoding_i_tensor = b_encodings_n[i].requires_grad_(True).to(model_device)
                grads = torch.autograd.grad(torch.sum(self.best_model.transformers[i](encoding_i_tensor)),
                                            encoding_i_tensor)
                feature_importance_i = (grads[0].abs().sum(0) / grads[0].abs().sum()).unsqueeze(0)
                if feature_importances.__contains__(skill):
                    feature_importances[skill] += feature_importance_i
                else:
                    feature_importances[skill] = feature_importance_i



        for i in range(self.skill_head.__len__()):
            skill = self.skill_head[i]
            encoding_i = np.vstack(encodings[skill]['encodings'])
            state_i = np.vstack(encodings[skill]['states'])

            if not os.path.exists("./StateData/" + skill + "/" + self.model_name):
                os.makedirs("./StateData/" + skill + "/" + self.model_name)

            savepath = "./StateData/" + skill + "/" + self.model_name + "/"
            encoding_df = pd.DataFrame(encoding_i,columns=self.encoded_behavior_heads)
            state_df = pd.DataFrame(state_i,columns=[skill+'_state'])
            encoding_df.to_csv(savepath+'encodings.csv',index=False)
            state_df.to_csv(savepath+'states.csv',index=False)

            encodings[skill]['encodings'] = encoding_df
            encodings[skill]['states'] = state_df

            # Compute feature importance via gradients
            feature_importance = feature_importances[skill]/self.sampled_usernum
            feature_importance_df = pd.DataFrame(feature_importance.detach().cpu().numpy(),columns=self.encoded_behavior_heads)
            feature_importance_df.to_csv(savepath + 'feaure_importances.csv', index=False)
            feature_importances[skill] = feature_importance_df

            feature_mean = self.best_model.encoding_norms[i].running_mean.unsqueeze(0)
            feature_mean_df = pd.DataFrame(feature_mean.detach().cpu().numpy(),columns=self.encoded_behavior_heads)
            feature_var = self.best_model.encoding_norms[i].running_var.unsqueeze(0)+self.best_model.encoding_norms[i].eps
            feature_var_df = pd.DataFrame(feature_var.detach().cpu().numpy(), columns=self.encoded_behavior_heads)
            feature_mean_df.to_csv(savepath + 'feaure_means.csv', index=False)
            feature_var_df.to_csv(savepath + 'feaure_vars.csv', index=False)
            encodings[skill]['feature_mean'] = feature_mean_df
            encodings[skill]['feature_var'] = feature_var_df
            print(skill + ": The encodings, states and feature importances are saved.")
        return encodings, feature_importances

    def load_encoded_and_state_data(self):
        print("Loading the encodings, states and feature importances...")
        feature_importances = {}
        encodings = {}
        for skill in self.skill_head:
            encodings[skill] = {}
            savepath = "./StateData/" + skill + "/" + self.model_name + "/"
            feature_importance_df = pd.read_csv(savepath + 'feaure_importances.csv')
            feature_importances[skill] = feature_importance_df

            encoding_df = pd.read_csv(savepath + 'encodings.csv')
            state_df = pd.read_csv(savepath + 'states.csv')
            encodings[skill]['encodings'] = encoding_df
            encodings[skill]['states'] = state_df
        return encodings, feature_importances

if __name__ == '__main__':
    irange = [1000]
    jrange = [2000]
    krange = [0]
    for i in irange:
        for j in jrange:
            for k in krange:
                tr = TransformerRegressor(
                    d_sequence_length=j,
                    d_score_min=0.01,
                    d_score_max=0.8,
                    m_att_hidden_num=i,
                    m_state_hidden_num=i,
                    m_att_droprate=0,
                    m_state_droprate=0,
                    m_encoder_reg=k,
                    max_epoch=50,
                    early_stop=10,
                    learning_rate=1e-4,
                    sampled_usernum = 1000,
                    device='cuda:0',
                )
                # tr.train()
                tr.load_best_model()
                # tr.save_errors()
                # tr.save_encoded_and_state_data(vis=True)
                # tr.save_encoded_and_state_data(vis=False)
                # encodings, feature_importances = tr.load_encoded_and_state_data()



    # tr = TransformerRegressor(
    #     d_sequence_length=5000,
    #     d_score_min=0.01,
    #     d_score_max=0.8,
    #     m_att_hidden_num=100,
    #     m_state_hidden_num=100,
    #     m_att_droprate=0,
    #     m_state_droprate=0,
    #     m_encoder_reg=1,
    #     max_epoch=100,
    #     early_stop=10,
    #     learning_rate=1e-4,
    #     sampled_usernum=300,
    #     device='cuda:0',
    # )
    # # tr.train()
    # tr.load_best_model()
    # tr.save_encoded_and_state_data()

