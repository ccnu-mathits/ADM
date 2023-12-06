# -*- coding: utf-8 -*-

"""

--------------------------------------

    Author: Xiaoxuan Shen

    Date:   2022/5/17 10:21

--------------------------------------

"""

import random
import numpy as np
import math
import json
import os



class Learning_Squences_Generator_double:
    def __init__(self,
                 alpha1 = 0.2,
                 alpha2 = 0.2,
                 beta1 = 0.2,
                 beta2 = 0.2,
                 gamma1 = 0.001,
                 gamma2 = 0.001,
                 theta = 0.2,
                 m_factor = 0.5,
                 learingrule1 = "Linear",
                 learingrule2 = "Linear",
                 correctness = "continuous",
                 n_exercise = 5,
                 n_student = 50,
                 n_length = 30,
                 ):
        # Fitness parameter in AL for skill 1 and 2 respectively
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        # Amount of improvement in BOTH MLP and AL for skill 1 and 2 respectively
        self.beta1 = beta1
        self.beta2 = beta2
        # The amount of forgetting for skill 1 and 2 respectively
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        # Standard deviation of noises
        self.theta = theta
        # The weight that skill 2 effect skill 1
        self.m_factor = m_factor
        #
        # Selection of preset learning rule from [Linear,Power,Exponential]
        self.learingrule1 = learingrule1
        self.learingrule2 = learingrule2

        # The form of the correctness from ["continuous","discrete"]
        # if TRUE: correctness is a discrete var from {0,1}
        # if FALSE: correctness is a continuous var from (0,1)
        self.correctness = correctness

        # Number of virtual exercises
        self.n_exercise = n_exercise
        # Number of virtual students
        self.n_student = n_student
        # Number of the length of learning sequence for each virtual student
        self.n_length = n_length
        # the path of saved Synthetic Data
        self.save_path = "./SyntheticData/" + self.learingrule1 + "+" + self.learingrule2 + "-" + self.correctness + '-' + \
               str(self.n_student) + 'stu' + str(self.n_length) + 'len' + str(self.n_exercise) + \
               "exer" + str(self.alpha1)+ "+" +  str(self.alpha2) + "a" + str(self.beta1) + "+" + str(self.beta2) + "b" + str(self.gamma1) + "+" +str(self.gamma2) + "g" +\
                         str(self.theta) +  "t.json"
        print("Generator is set...")


    # exercise generator
    # {id:(diff,(Q))...}, parm_matrix = numpy.array([n_exercise,1])
    # diff means the difficulty of the exercise where diff follows a uniform distribution
    # diff from (0,1)
    def generate_exercises(self, diff_distribution='uniform'):
        exercises = {}
        # generate exercises for skill 1
        for i in range(self.n_exercise):
            if diff_distribution == 'uniform':
                diff = (random.random(),0)
                Q = (1,0)
                exercises[i] = (diff,Q)
            else:
                print("Wrong diff_distribution!!!")
                quit()
        # generate exercises for skill 2
        for j in range(self.n_exercise):
            if diff_distribution == 'uniform':
                diff = (0,random.random())
                Q = (0,1)
                exercises[j+self.n_exercise] = (diff,Q)
            else:
                print("Wrong diff_distribution!!!")
                quit()
        return exercises

    # student generator
    # {id:(init_state),...}
    def generate_students(self, init_state_distribution='uniform'):
        students = {}
        for i in range(self.n_student):
            students[i] = (0.0, 0.0)
        return students

    def Linear_learner(self,init_state, N, N_total, beta, gamma):
        return init_state + beta * N - gamma * N_total

    def Power_learner(self,init_state, N, N_total, alpha,beta, gamma):
        return init_state + beta*(N**(alpha)) - gamma*N_total

    def Exponential_learner(self,init_state, N, N_total,alpha,beta, gamma):
        return init_state + beta*(math.exp(alpha*N)) - gamma*N_total

    def IRT_model(self,st,diff,Q,guess = 0.1):
        if Q[0] == 1:
            return guess + (1 - guess) / (1 + math.exp(-1.702 * (st[0] - diff[0])))
        elif Q[1] == 1:
            return guess + (1 - guess) / (1 + math.exp(-1.702 * (st[1] - diff[1])))
        else:
            print("Wrong Q matrix for IRT model...")
            quit()

    def get_exercises(self,exercises_list):
        return random.choice(exercises_list)

    def get_correctness(self,p):
        if self.correctness == "discrete":
            sample = random.random()
            if sample < p:
                return 1
            else:
                return 0
        elif self.correctness == "continuous":
            return p
        else:
            print("Wrong correctness form!!!")
            quit()


    # learning_squences = {uid,{params:{init_state:x,alpha_i:x,beta_i:x,gamma_i:x}
    #                           q_squences:[qid,qid,...],           n_length
    #                           diff_squences:[diff,diff,...],      n_length
    #                           skill_squences:[Q,Q,...],           n_length
    #                           state_squences:[state,state,...],   n_length+1
    #                           correctness_squences:[0,0,1,...],   n_length
    #                           }}
    #
    def generate_learning_squences(self,students,exercises):
        learning_squences = {}
        exercises_list = list(exercises.items())
        max_state1 = 0
        max_state2 = 0
        for uid in students:
            init_state = students[uid]
            learning_squences[uid] = {}
            learning_squences[uid]["params"] = {}
            learning_squences[uid]["params"]["init_state"] = init_state
            learning_squences[uid]["params"]["alpha1"] = self.alpha1
            learning_squences[uid]["params"]["alpha2"] = self.alpha2
            learning_squences[uid]["params"]["beta1"] = self.beta1
            learning_squences[uid]["params"]["beta2"] = self.beta2
            learning_squences[uid]["params"]["gamma1"] = self.gamma1
            learning_squences[uid]["params"]["gamma2"] = self.gamma2

            q_squences = []
            state_squences = [init_state]
            N = np.zeros(2)

            for l in range(self.n_length):
                q, q_feature = self.get_exercises(exercises_list)
                q_squences.append(q)
                N += np.array(q_feature[1])

                # the N of skill 1 equals (1-m_factor)*N1+*m_factorN2 because the M theory
                # the N of skill 2 equals N2
                N1 = (1-self.m_factor) * N[0] + self.m_factor * N[1]
                N2 = N[1]
                N_total = N[1] + N[0]
                init_state1 = init_state[0]
                init_state2 = init_state[1]
                st = state_squences[state_squences.__len__() - 1]

                if self.learingrule1 == "Linear":
                    new_st1 = self.Linear_learner(init_state1, N1, N_total, self.beta1, self.gamma1)
                elif self.learingrule1 == "Power":
                    new_st1 = self.Power_learner(init_state1, N1, N_total, self.alpha1, self.beta1, self.gamma1)
                elif self.learingrule1 == "Exponential":
                    new_st1 = self.Exponential_learner(init_state1, N1, N_total, self.alpha1, self.beta1, self.gamma1)
                else:
                    print("Wrong learing rule!!!")
                    quit()

                if self.learingrule2 == "Linear":
                    new_st2 = self.Linear_learner(init_state2, N2, N_total, self.beta2, self.gamma2)
                elif self.learingrule2 == "Power":
                    new_st2 = self.Power_learner(init_state2, N2, N_total, self.alpha2, self.beta2, self.gamma2)
                elif self.learingrule2 == "Exponential":
                    new_st2 = self.Exponential_learner(init_state2, N2, N_total, self.alpha2, self.beta2, self.gamma2)
                else:
                    print("Wrong learing rule!!!")
                    quit()

                new_st = (new_st1,new_st2)
                state_squences.append(new_st)
                if new_st1>max_state1:
                    max_state1 = new_st1
                if new_st2>max_state2:
                    max_state2 = new_st2
                learning_squences[uid]["q_squences"] = q_squences
                learning_squences[uid]["state_squences"] = state_squences

        #update the diff of questions for adapting the learner's level
        for e in exercises:
            exercises[e] = ((exercises[e][0][0]*max_state1,exercises[e][0][1]*max_state2),exercises[e][1])

        for uid in students:
            diff_squences = []
            skill_squences = []
            correctness_squences = []
            for i in range(self.n_length):
                qid = learning_squences[uid]["q_squences"][i]
                diff = exercises[qid][0]
                skill = exercises[qid][1]
                diff_squences.append(diff)
                skill_squences.append(skill)
                st = learning_squences[uid]["state_squences"][i]
                correctness = self.get_correctness(self.IRT_model(st, diff, skill)) + random.gauss(0,self.theta)
                correctness_squences.append(correctness)
                learning_squences[uid]["diff_squences"] = diff_squences
                learning_squences[uid]["skill_squences"] = skill_squences
                learning_squences[uid]["correctness_squences"] = correctness_squences
        return learning_squences


    def save_syntheic_data(self,students,exercises,squences):
        path = self.save_path
        with open(path, "w") as f:
            json.dump({"students":students,"exercises":exercises,"squences":squences}, f)
            f.close()
            print("The syntheic data is dumped in:")
            print(path)

    def load_syntheic_data(self):
        path = self.save_path
        with open(path,"r") as f:
            all = json.load(f)
            students = all["students"]
            exercises = all["exercises"]
            squences = all["squences"]
            return students,exercises,squences

    def generate_syntheic_dataset(self):
        print("Checking the dumps...")
        if os.access(self.save_path, os.F_OK):
            print("Syntheic dataset is existent!")
            return self.load_syntheic_data()
        else:
            students = self.generate_students()
            exercises = self.generate_exercises()
            squences = self.generate_learning_squences(students, exercises)
            self.save_syntheic_data(students,exercises,squences)
            return students,exercises,squences

    def states_vis(self, squences):
        states = []
        for i in range(self.n_student):
            states.append(
                np.array(squences[i]["state_squences"])
            )
        states_array = np.vstack(states)
        np.savetxt("states.csv",states_array,delimiter=",")
        print("aaa")
        return states_array



if __name__ == '__main__':

    g = Learning_Squences_Generator_double(
        alpha1=1,
        alpha2=0.1,
        beta1=0.3,
        beta2=1,
        gamma1=0.1,
        gamma2=0.1,
        theta=0,
        m_factor=0.3,
        learingrule1="Linear",      #[Linear,Power,Exponential]
        learingrule2="Exponential",      #[Linear,Power,Exponential]
        correctness="continuous",
        # correctness="discrete",
        n_exercise=20,
        n_student=30,
        n_length=20,
    )
    students = g.generate_students()
    exercises = g.generate_exercises()
    squences = g.generate_learning_squences(students, exercises)
    # students, exercises, squences = g.generate_syntheic_dataset()