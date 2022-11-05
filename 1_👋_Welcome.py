# setup
import pickle
import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie
import requests
# website build
# header of the welcome page
st.markdown('''<h1 style='text-align: center; color: #FF4B4B;'> WELCOME!</h1>''', unsafe_allow_html=True)
st.write('---')

st.write('\n')
st.write('\n')
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lotti_ur = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_hvmn385j.json')
# second grid for animated image   
# animated image loading function 

col1, col2 =st.columns([1,1])

with col1:
    st.write('\n')
    
    st.caption('''<h6 
            style='text-align: left; 
            color: #fffff;'> 
            Hi!👋 This is an IPL match win Probability Predictor Web Application. By using Machine Learning 
            I build this Web Application. It is capable of predicting the probability of any ongoing match Winner. 
            By taking some information about current scenario of the match as an input. After providing the information 
            required to predict the probability. It will show the Probability of win the match for both teams based on all 
            previous record.</h6>''',unsafe_allow_html=True)
    st.write('\n')
    st.caption('''<h6 
            style='text-align: left; 
            color: #fffff;'>My vision for build this Web Application is to predict the winner before end of any match
            , don't misuse of this Web Application.
\nThank You!</h6>''',unsafe_allow_html=True)
            
with col2:
    st_lottie(lotti_ur,
          speed = 1,
          key=None,
          height=200,
          width=380,
          quality='high',
          reverse=False
          )

st.write('---')
st.caption('''<h6 
            style='text-align: center; 
            color: #fffff;'> 
            The Indian Premier League (IPL), is a men's T20 franchise cricket league of India. It is annually contested by eight 
            teams. The league was founded by the Board of Control for Cricket in India (BCCI) in 2007. Brijesh Patel is the incumbent chairman of IPL. It is usually held annually in summer across India between March to May 
            and has an exclusive window in the ICC Future Tours Programme. </h6>''',unsafe_allow_html=True)

