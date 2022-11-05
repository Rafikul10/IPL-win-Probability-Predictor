# setup
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from annotated_text import annotated_text
import base64
# ML model load
LR_model = pickle.load(open('LR_pipe.pkl','rb'))    # LinearRegression Model for probability calculate
RFC_model = pickle.load(open('RFC_pipe.pkl','rb'))  # RandomForestClassifier for Winnenr Prediction

# web page
st.markdown('''<h2 
            style='text-align: center; 
            color: #FF4B4B;'> 
            IPL Win Probability Predictor
            </h2>''', 
            unsafe_allow_html=True)
st.caption('''In order to use the Application provide all the details of any ongoing match and hit the predict button to see the result.
            If you don't know the venue of the match just choose any one option from dropdown box.''')
st.write('---')
st.write('\n')
st.write('\n')
# user input box
teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']
 
city = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Batting team',sorted(teams))
with col2:
    bolwling_team = st.selectbox('Bowling team', sorted(teams))
    
city = st.selectbox('Select Venue', sorted(city))

col3, col4 = st.columns(2)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    wickets_out = st.number_input('Wickets out')

col5, col6 = st.columns(2)

with col5:
    overs_completed = st.number_input('Overs completed')
with col6:
    total_runs_x = st.number_input('Target', ) 

st.write('\n')
_, col7, _ = st.columns([1.37,1,1])

with col7:
    predict_button = st.button('Predict Winner')


if predict_button:
    if batting_team == bolwling_team:
        st.warning('Invalid information!')
    elif total_runs_x  == 0:
        st.warning('Invalid score')
    elif current_score == 0:
        st.warning('Invalid score')
    elif overs_completed == 0:
        st.warning('Invalid score')
        
        
    else:
        runs_left = total_runs_x - current_score
        balls_left = 120 - overs_completed*6
        wickets_left = 10 - wickets_out
        crr = current_score / overs_completed 
        rrr = (runs_left*6) / balls_left
        
        # dataframe of user input
        input_df = pd.DataFrame({'batting_team':[batting_team],
                         'bowling_team':[bolwling_team],
                         'city':[city],
                         'runs_left':[runs_left],
                         'balls_left':[balls_left],
                         'wickets_left':[wickets_left],
                         'total_runs_x':[total_runs_x],
                         'crr':[crr],
                         'rrr':[rrr]})
  
        # LinearRegression Model    
        result = LR_model.predict_proba(input_df)
        loss = result[0][0]
        winner = result[0][1]
        _, col1, _ = st.columns([0.5,1.5,0.5])
        with col1:
            st.markdown('''<h2 
            style='text-align: center; 
            color: #FF4B4B;'> 
            Win Probability
            </h2>''', 
            unsafe_allow_html=True)
            st.write('---')        
        st.write('\n')
        
        col1, col2, col3  = st.columns([0.9,0.4,0.9])
        with col1:
            # iamge show
            if batting_team == 'Sunrisers Hyderabad':
                st.image('https://i.ibb.co/DRMwCms/SH.png')
            elif batting_team == 'Mumbai Indians':
                st.image('https://i.ibb.co/PTdF2rY/mi-new.png')
            elif batting_team == 'Royal Challengers Bangalore':
                st.image('https://i.ibb.co/pnVmVxB/RCB.png')
            elif batting_team == 'Kolkata Knight Riders':
                st.image('https://i.ibb.co/2PBVmRK/KKR.png')
            elif batting_team == 'Kings XI Punjab':
                st.image('https://i.ibb.co/M8FCB1t/KP.png')
            elif batting_team == 'Chennai Super Kings':
                st.image('https://i.ibb.co/Cthrw2r/CSK.png')
            elif batting_team == 'Rajasthan Royals':
                st.image('https://i.ibb.co/CvZnNyM/RR.png')
            elif batting_team == 'Delhi Capitals':
                st.image('https://i.ibb.co/RCtmMpj/DC.png')
            
            
        with col2:
            # st.write('''Vs''')
            st.image('https://i.ibb.co/vvw904g/vs-new.png')
            
        with col3:    
            if bolwling_team == 'Sunrisers Hyderabad':
                st.image('https://i.ibb.co/DRMwCms/SH.png')
            elif bolwling_team == 'Mumbai Indians':
                st.image('https://i.ibb.co/PTdF2rY/mi-new.png')
            elif bolwling_team == 'Royal Challengers Bangalore':
                st.image('https://i.ibb.co/pnVmVxB/RCB.png')
            elif bolwling_team == 'Kolkata Knight Riders':
                st.image('https://i.ibb.co/2PBVmRK/KKR.png')
            elif bolwling_team == 'Kings XI Punjab':
                st.image('https://i.ibb.co/M8FCB1t/KP.png')
            elif bolwling_team == 'Chennai Super Kings':
                st.image('https://i.ibb.co/Cthrw2r/CSK.png')
            elif bolwling_team == 'Rajasthan Royals':
                st.image('https://i.ibb.co/CvZnNyM/RR.png')
            elif bolwling_team == 'Delhi Capitals':
                st.image('https://i.ibb.co/RCtmMpj/DC.png')
            
        
        _, col1, _, col2, _ = st.columns([0.35,0.3,1.3,0.4,0.2])
        with col1:
            st.title(str(round(winner*100)) +'%')
            st.write('---')
            
        with col2:
            st.title( str(round(loss*100)) +'%')
            st.write('---')

        # RandomForestClassifier model
        # result = RFC_model.predict(input_df)
        # if result ==1:
        #     st.write('Winner will be  - ' + batting_team) 
        # else:
        #     st.write('Wiiner will be - ' + bolwling_team)       
        
    