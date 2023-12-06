# -*- coding: utf-8 -*-

"""

--------------------------------------

    Author: Xiaoxuan Shen

    Date:   2023/2/7 14:53

--------------------------------------

"""
import torch
from DataGenerator import Learning_Squences_Generator_double
from TorchModels import Trasnsformer_model
from torch.optim import lr_scheduler
import numpy as np
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter
import warnings
import os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class TransformerRegressor:
    def __init__(self,
                 g_alpha1=0.2,
                 g_alpha2=0.2,
                 g_beta1=0.4,
                 g_beta2=0.6,
                 g_gamma1=0.2,
                 g_gamma2=0.2,
                 g_learingrule1="Linear",  # [Probabilistic,Linear,Power,Exponential]
                 g_learingrule2="Linear",
                 g_theta=0.0,
                 g_m_factor=0.5,
                 g_correctness="continuous",  # [continuous,discrete]
                 g_n_exercise=5,
                 g_n_student=20,
                 g_n_length=15,
                 m_att_hidden_num = 50,
                 m_state_hidden_num = 50,
                 m_att_reg = 1,
                 max_epoch=5000,
                 early_stop=500,
                 learning_rate=0.001,
                 device='cpu',
                 ):
        self.n_exercise = g_n_exercise
        self.n_student = g_n_student
        self.n_length = g_n_length
        self.g_theta = g_theta

        self.g_alpha1 = g_alpha1
        self.g_alpha2 = g_alpha2
        self.g_beta1 = g_beta1
        self.g_beta2 = g_beta2
        self.g_gamma1 = g_gamma1
        self.g_gamma2 = g_gamma2
        self.g_learingrule1 = g_learingrule1
        self.g_learingrule2 = g_learingrule2
        self.g_correctness = g_correctness

        self.m_att_hidden_num = m_att_hidden_num
        self.m_state_hidden_num = m_state_hidden_num
        self.m_att_reg = m_att_reg

        self.max_epoch = max_epoch
        self.early_stop = early_stop
        self.lr = learning_rate

        self.model_path = self.g_learingrule1 + "+" + self.g_learingrule2 + "-" + self.g_correctness + "-" + str(self.g_theta) + \
                          '-H' + str(self.m_att_hidden_num) + "-"+str(self.m_state_hidden_num)+'-reg'+str(self.m_att_reg)

        self.writer = SummaryWriter(log_dir=('runs/' + self.model_path))

        self.device = device

        g = Learning_Squences_Generator_double(
            alpha1=g_alpha1,
            alpha2=g_alpha2,
            beta1=g_beta1,
            beta2=g_beta2,
            gamma1=g_gamma1,
            gamma2=g_gamma2,
            theta=g_theta,
            m_factor=g_m_factor,
            learingrule1=g_learingrule1,
            learingrule2=g_learingrule2,
            correctness=g_correctness,
            n_exercise=g_n_exercise,
            n_student=g_n_student,
            n_length=g_n_length,
        )
        students, exercises, squences = g.generate_syntheic_dataset()
        self.dataset = [squences, students, exercises]
        self.QustionS = []
        self.ContextS = []
        self.CorrectS = []
        self.StateS = []
        self.Q_infos = []
        for i in squences:
            skill_sq = torch.tensor(squences[i]["skill_squences"])
            if skill_sq.ndim == 1:
                skill_sq = skill_sq.unsqueeze(1)
            diff_sq = torch.tensor(squences[i]["diff_squences"])
            if diff_sq.ndim == 1:
                diff_sq = diff_sq.unsqueeze(1)
            self.Q_infos.append(
                torch.cat([
                    diff_sq,skill_sq
                ], 1).to(self.device)
            )
            self.QustionS.append(
                torch.cat([
                    torch.ones(skill_sq.size()[0],1),
                    skill_sq
                ], 1).to(self.device)
            )
            q_sq = squences[i]["q_squences"]
            q_count = {}
            q_index = {}
            count = []
            interval = []
            index = 0
            for ii in q_sq:
                index += 1
                if q_count.__contains__(ii):
                    count.append(q_count[ii])
                    q_count[ii] += 1
                else:
                    count.append(0)
                    q_count[ii] = 1
                if q_index.__contains__(ii):
                    interval.append(index-q_index[ii])
                else:
                    interval.append(0)
                q_index[ii] = index

            self.ContextS.append(
                torch.cat([
                    torch.tensor(count).unsqueeze(1),
                    torch.tensor(interval).unsqueeze(1),
                ], 1).to(self.device)
            )
            self.CorrectS.append(
                torch.cat([
                    torch.tensor(squences[i]["correctness_squences"]).unsqueeze(1),
                ], 1).to(self.device)
            )
            state_sq = torch.tensor(squences[i]["state_squences"])
            if state_sq.ndim == 1:
                state_sq = state_sq.unsqueeze(1)
            self.StateS.append(
                torch.cat([
                    state_sq,
                ], 1).to(self.device)
            )

    def train(self):
        self.model = Trasnsformer_model(
            D_Q=self.QustionS[0].size()[1],
            D_C=self.ContextS[0].size()[1],
            D_S=self.StateS[0].size()[1],
            ATT_HIDDEN_SIZE=self.m_att_hidden_num,
            MAX_LENGTH=self.n_length,
            STATE_HIDDEN_SIZE=self.m_state_hidden_num,
            DEVICE=self.device
        ).to(self.device)
        optimizer = opt.Adam(self.model.parameters(), lr=self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

        best_loss = 10
        stop = 0

        for e in range(self.max_epoch):
            self.model.train()
            fit_loss = 0
            att_reg = 0
            for s in range(self.n_student):
                questions = self.QustionS[s]
                contexts = self.ContextS[s]
                correctness = self.CorrectS[s]
                q_info = self.Q_infos[s]
                _, _, _, _, _, att_all, p_correctness = self.model.forward(questions, contexts,q_info)
                score_fit_loss = (correctness - p_correctness).square().mean()
                if torch.isnan(score_fit_loss):
                    print("Loss value NAN:",score_fit_loss)
                    quit()
                fit_loss += score_fit_loss.detach()
                att_smooth_loss = (att_all - 1).square().mean()
                att_reg += att_smooth_loss.detach()
                total_loss = score_fit_loss + self.m_att_reg * att_smooth_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            train_fit_loss = (fit_loss / self.n_student).cpu().numpy()
            train_att_reg = (att_reg / self.n_student).cpu().numpy()
            train_total_loss = train_fit_loss + self.m_att_reg * train_att_reg
            scheduler.step()
            print('epoch:', e, '| Total_loss:', round(train_total_loss.tolist(),8), '| Fit_loss:', round(train_fit_loss.tolist(),8), '| Att_reg:',
                  round(train_att_reg.tolist(),8))
            self.writer.add_scalar('Fit_loss', train_fit_loss, e)
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
        torch.save(self.model, "./SavedModels/"+self.model_path+".pth")
        print("Model is saved in:")
        print("./SavedModels/"+self.model_path+".pth")

    def load_bast_model(self):
        self.best_model = torch.load("./SavedModels/"+self.model_path+".pth")
        print("The best model is loaded from:")
        print("./SavedModels/"+self.model_path+".pth")

    def save_data_in_state_MLP(self):
        self.best_model.eval()
        b_encodings_1 = []
        b_encodings_1_n = []
        b_encodings_2 = []
        b_encodings_2_n = []
        p_states_1 = []
        p_states_2 = []
        for s in range(self.n_student):
            questions = self.QustionS[s]
            contexts = self.ContextS[s]
            q_info = self.Q_infos[s]
            _, _, b_encodings, b_encodings_n, p_states, _, _ = self.best_model.forward(questions, contexts, q_info)
            b_encodings_1_n.append(b_encodings_n[0])
            b_encodings_2_n.append(b_encodings_n[1])
            b_encodings_1.append(b_encodings[0].cpu().detach().numpy())
            b_encodings_2.append(b_encodings[1].cpu().detach().numpy())
            p_states_1.append(p_states[0].cpu().detach().numpy())
            p_states_2.append(p_states[1].cpu().detach().numpy())
        b_encodings_1 = np.vstack(b_encodings_1)
        b_encodings_2 = np.vstack(b_encodings_2)
        p_states_1 = np.vstack(p_states_1)
        p_states_2 = np.vstack(p_states_2)

        encoding_1_tensor = b_encodings_n[0].requires_grad_(True)
        grads = torch.autograd.grad(torch.sum(self.best_model.state1_MLP(encoding_1_tensor)),
                                    encoding_1_tensor)
        feature_importance_1 = grads[0].abs().sum(0)*encoding_1_tensor.mean(0)
        feature_importance_1 = (feature_importance_1/feature_importance_1.sum()).cpu().detach().numpy()

        encoding_2_tensor = b_encodings_n[1].requires_grad_(True)
        grads = torch.autograd.grad(torch.sum(self.best_model.state2_MLP(encoding_2_tensor)),
                                    encoding_2_tensor)
        feature_importance_2 = grads[0].abs().sum(0)*encoding_2_tensor.mean(0)
        feature_importance_2 = (feature_importance_2/feature_importance_2.sum()).cpu().detach().numpy()

        print("Feature importance for skill 1:",feature_importance_1)
        print("Feature importance for skill 2:",feature_importance_2)

        savepath = "./StateMLPData/" + self.model_path + "_b_encoding1.csv"
        np.savetxt(savepath, b_encodings_1, delimiter=',')
        savepath = "./StateMLPData/" + self.model_path + "_b_encoding2.csv"
        np.savetxt(savepath, b_encodings_2, delimiter=',')
        savepath = "./StateMLPData/" + self.model_path + "_states1.csv"
        np.savetxt(savepath, p_states_1, delimiter=',')
        savepath = "./StateMLPData/" + self.model_path + "_states2.csv"
        np.savetxt(savepath, p_states_2, delimiter=',')
        savepath = "./StateMLPData/" + self.model_path + "_importance1.csv"
        np.savetxt(savepath, feature_importance_1, delimiter=',')
        savepath = "./StateMLPData/" + self.model_path + "_importance2.csv"
        np.savetxt(savepath, feature_importance_2, delimiter=',')
        print("The input, output, and importances are saved...")

        return feature_importance_1, feature_importance_2

if __name__ == '__main__':
    i_range = [0]
    j_range = [1]
    k_range = [1000]
    for i in i_range:
        for j in j_range:
            for k in k_range:
                tr = TransformerRegressor(
                    g_alpha1=0.1,
                    g_alpha2=0.8,
                    g_beta1=1,
                    g_beta2=1,
                    g_gamma1=0.1,
                    g_gamma2=0.2,
                    g_learingrule1="Exponential",  # [Linear,Power,Exponential]
                    g_learingrule2="Power",  # [Linear,Power,Exponential]
                    g_u_theta=i,
                    g_m_factor=0.3,
                    g_correctness="continuous",  # [continuous,discrete]
                    g_n_exercise=20,
                    g_n_student=50,
                    g_n_length=20,
                    m_att_hidden_num=100,
                    m_state_hidden_num=k,
                    m_att_reg=j,
                    early_stop=5000,
                    learning_rate=5e-4,
                    device="cuda:0"
                )
                # tr.train()
                # tr.load_bast_model()
                # tr.save_data_in_state_MLP()



