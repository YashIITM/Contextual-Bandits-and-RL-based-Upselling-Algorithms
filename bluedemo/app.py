
import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
from copy import deepcopy
from src.env import SampleContext, GetRealReward
from tqdm import tqdm
import altair as alt
import torch

st.header('Blue Action 3.0 POC')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
type(device)

np.random.seed(12345)
col1, col2 = st.columns(2)
K = col1.text_input('number of action', value=10)
K =  int(K)
T = 6000 
d = col1.text_input('customer state length', value=10)
d =  int(d)
L = 2
m = 30 
gamma_t = 0.01
nu = 0.1 
lambda_ = 1
delta = 0.01 
S = 0.01 
eta = 1e-3 
frequency = col2.text_input('Update Frequency', value=100)
frequency = int(frequency)
batchsize = 100
verbose = False


from src.neuralucb import NeuralAgent
neuralagent = NeuralAgent(
    K=K, T=T, d=d, L=L, m=m, gamma_t=gamma_t,
    nu=nu, lambda_=lambda_, delta=delta,
    S=S, eta=eta, frequency=frequency,
    batchsize=batchsize
)

import pandas as pd
from datetime import datetime
dates_ = pd.date_range(start = datetime.today(), periods = 100000, freq='H').to_pydatetime().tolist()
dates_ = pd.to_datetime(dates_)



def plot_animation(df):
    lines = alt.Chart(df).mark_line().encode(
       x=alt.X('date:T' ,axis=alt.Axis(title='Customer Feed Intake Data')),
       y=alt.Y('rank:Q', axis=alt.Axis(
        title='<- Next Worst Action                              %Time NBA is Selected                              Next Best Action -> ',)),
     ).properties(
       width=900,
       height=600
     ) 
    return lines


action_list = []
best_action = []
action_rank = []
df = pd.DataFrame()
df['date'] = [dates_[0]]
df['rank'] = [0]

lines = alt.Chart(df).mark_line().encode(
  x=alt.X('1:T',axis=alt.Axis(title='Customer Feed Intake Data')),
  y=alt.Y('0:Q',axis=alt.Axis( 
  title=' <-Next Worst Action               Next Best Action -> '))
  ).properties(
      width=900,
      height=600
  )

import pickle
A = np.random.normal(loc=0, scale=1, size=(d, d))
with open('Reward_Matrix.pkl', 'wb') as f:
    pickle.dump(A, f)


start_btn = col2.button('Start Blue Action POC')



if start_btn:
    line_plot = st.altair_chart(lines)
    place_holders = {}
    for i in range(K):
        place_holders[i] = st.empty()

    for tt in tqdm(range(1, T + 1)):

        with open('Reward_Matrix.pkl', 'rb') as f:
            A = pickle.load(f)
        
        context_list = SampleContext(d, K)
        realized_reward = GetRealReward(context_list, A)
        best_action_for_customer = np.argmax(realized_reward)
        best_action.append(best_action_for_customer)
        action_order = np.flip(np.argsort(realized_reward))
        neural_ind = neuralagent.Action(context_list)
        action_list.append(neural_ind)
        action_rank.append(list(action_order).index(neural_ind))
        neural_reward = realized_reward[neural_ind]
        neuralagent.Update(neural_reward)

        df = pd.DataFrame()
        df['date'] = dates_[:len(action_rank)]
        df['rank'] = K - np.array(action_rank)
        df['rank'] = df['rank'].ewm(com=20).mean()
        df['rank'] = df['rank']*10


        lines = plot_animation(df)
        line_plot.altair_chart(lines)

        df_actions = pd.DataFrame()
        df_actions['action_agent'] = action_list
        df_actions['action_best'] = best_action[:len(action_list)]

 
        df_actions_last_100 = df_actions[-50:]
        from collections import Counter
        if len(df_actions) > 100:
            nba = {}
            for i in range(K):
                try:
                    nba[i] = 100*Counter(df_actions_last_100[df_actions_last_100['action_best'] == i]['action_agent'].values)[i]/len(df_actions_last_100[df_actions_last_100['action_agent'] == i])
                    
                    if nba[i] > 75:
                        place_holders[i].success('When NBA is action: '+ str(i) + ' , ' + str(int(nba[i])) + '% times action ' + str(i)+ ' is choosen')
                    else:
                        place_holders[i].warning('When NBA is action: '+ str(i) + ' , ' + str(int(nba[i])) + '% times action ' + str(i)+ ' is choosen')
                except:
                    a =1
