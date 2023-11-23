import pickle 
import streamlit as st
import numpy as np

with open('Reward_Matrix.pkl', 'rb') as f:
    A = pickle.load(f)

randomness = st.slider('Randomness Introduction',min_value=0, max_value=100,value=50)
if st.button('Simulate Change in Environment'):
    A = A + np.random.normal(loc=0, scale=randomness/100, size=(A.shape[0], A.shape[0]))  
    with open('Reward_Matrix.pkl', 'wb') as f:
        pickle.dump(A, f)
