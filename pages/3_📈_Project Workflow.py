

import streamlit as st
import streamlit as st
from streamlit_lottie import st_lottie
import requests
import extra_streamlit_components as stx
# heading of the page 
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'> Project Workflow</h1>", unsafe_allow_html=True)
st.write('\n')
# columns for shields 
_ ,col2,_= st.columns([0.1,2,0.1])

with col2:
    st.write('''[![Data kaggle](https://img.shields.io/badge/Data-Kaggle-blueviolet)](https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set) 
             [![scikitlearn](https://img.shields.io/badge/Scikit--learn-1.0.2-orange)](https://scikit-learn.org/stable/tutorial/index.html) 
             [![Python 3.10.0](https://img.shields.io/badge/Python-3.10.0-brightgreen)](https://www.python.org/downloads/release/python-3100/) 
             [![Github](https://camo.githubusercontent.com/3a41f9e3f8001983f287f5447462446e6dc1bac996fedafa9ac5dae629c2474f/68747470733a2f2f62616467656e2e6e65742f62616467652f69636f6e2f4769744875623f69636f6e3d67697468756226636f6c6f723d626c61636b266c6162656c)](https://github.com/Rafikul10/IPL-win-Probability-Predictor) 
             [![Streamlit 1.14.0](https://img.shields.io/badge/Streamlit%20-1.14.0-Ff0000)](https://docs.streamlit.io/) 
             [![Cloud Platform](https://img.shields.io/badge/CloudPlatform-Heroku-9cf)](https://www.heroku.com/managed-data-services)''')

st.write('-----')

# animated image loading function 
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# columns for animated image and description of page
col1, col2 = st.columns(2)
with col1:
    st.markdown("<h3 style='text-align: center; color: #fffff;'> Description</h3>", unsafe_allow_html=True)
    st.caption(f'''IPL win Probability Predictor System is a simple Machine Learning Project.
               By collecting the data from [Kaggle](https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set).
               I started the project and then I 
               did some preprocessing on the dataset and build the final model. 
               It takes few information about ongoing matches as an input and predict the win
               probability for both teams. For predicting the probability I used LogisticRegression algorithm. 
               For more information scroll down 
               and check out the Model Build page. Output of the code is not available 
               write all code in your notebook and run to see the output!
                
               ''')

# url of animated image
lotti_ur = load_lottieurl('https://assets6.lottiefiles.com/packages/lf20_w51pcehl.json')
# second grid for animated image   
with col2:
    st_lottie(lotti_ur,
          speed = 1,
          key=None,
          height=280,
          width=350,
          quality='high',
          reverse=False
          )

         
st.write('-----')
st.markdown("<h2 style='text-align: center; color: #FF4B4B;'>  Steps </h2>", unsafe_allow_html=True)

val = stx.stepper_bar(steps=["DataCollection🗂️", "Preprocessing👨‍💻", "Model Build🤖",'Website Build🌐','Deployment🎯'])

if val == 0:
    st.write('----')
    st.markdown("<h2 style='text-align: center; color: #FF4B4B;;'> Data Collection Processs</h2>", unsafe_allow_html=True)
    # columns create for align the text in middle
    col1, col2, col3 = st.columns([0.35,1,0.1])
    with col1:
        pass
    with col2:
        st.write('''[![Data Kaggle](https://img.shields.io/badge/Data-Kaggle-blueviolet)](https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set)
                 [![Size](https://img.shields.io/badge/Size-45.74mb-br)](https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set)
                 [![File Format](https://img.shields.io/badge/FileFormat-.csv-blue)](https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set)
                 [![Github](https://camo.githubusercontent.com/3a41f9e3f8001983f287f5447462446e6dc1bac996fedafa9ac5dae629c2474f/68747470733a2f2f62616467656e2e6e65742f62616467652f69636f6e2f4769744875623f69636f6e3d67697468756226636f6c6f723d626c61636b266c6162656c)](https://github.com/Rafikul10/IPL-win-Probability-Predictor)''')
    
    st.write('\n')
    st.markdown(f'''<h4 style = 'text-align: left;'> Dependencies :</h4>''',
                    unsafe_allow_html=True)
    st.markdown(f'''* Jupyter Notebook''') 
    st.markdown(f'''* Python 3.10.0''')
    st.markdown(f'''* Pandas''')
    st.markdown(f'''* Numpy''')
    st.caption(f'''Install dependencies using [conda](https://docs.conda.io/en/latest/)''')
    
    st.markdown(f'''<h4 style = 'text-align: left;'> ⚙️Setup :</h4>''',
                    unsafe_allow_html=True)
    st.code('''import numpy as np
import pandas as pd''')
    
    st.markdown(f'''<h4 style = 'text-align: left;'> 🗂️Dataset import :
 
>Two different dataset is there deliveries and matches imported both the dataset for preprocessing.:</h4>''',
                    unsafe_allow_html=True)
    st.code('''delivery = pd.read_csv('deliveries.csv')    # import deliveries data
match = pd.read_csv('matches.csv)           # import matches data''')
    st.code('match.shape    # checkc shape of match dataset')
    st.code('match.head()   # check match dataset')
    st.code('delivery.head()  # check delivery dataset')
    
    st.write('\n')
    
    st.write('**Note** : Data collection is done now for further process check 2nd step - **Preprocessing** from top.')
    st.write('\n')
    st.write('\n')
    st.caption('''👨‍💻For full code with output check my [GitHub](https://github.com/Rafikul10/IPL-win-Probability-Predictor) repositories.''')   

# now preprocessing page
if val == 1:
    st.write('---')
    st.markdown("<h2 style='text-align: center; color: #FF4B4B;;'> Data Preprocessing</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.32,1,0.1])
    with col1:
        pass
    with col2:
        st.write('''[![Data Kaggle](https://img.shields.io/badge/Data-Kaggle-blueviolet)](https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set)
                 [![Github](https://camo.githubusercontent.com/3a41f9e3f8001983f287f5447462446e6dc1bac996fedafa9ac5dae629c2474f/68747470733a2f2f62616467656e2e6e65742f62616467652f69636f6e2f4769744875623f69636f6e3d67697468756226636f6c6f723d626c61636b266c6162656c)](https://github.com/Rafikul10/IPL-win-Probability-Predictor)
                 [![Python 3.10.0](https://img.shields.io/badge/Python-3.10.0-brightgreen)](https://www.python.org/downloads/release/python-3100/)
                 [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-red)](https://scikit-learn.org/stable/)''')
                 
    with col3:
        pass
    st.write('\n')
    st.markdown(f'''<h4 style = 'text-align: left;'> Dependencies :</h4>''',
                    unsafe_allow_html=True)
    st.markdown(f'''* Jupyter Notebook''') 
    st.markdown(f'''* Python 3.10.0''')
    st.markdown(f'''* scikit-learn 1.0.2''')
    st.markdown(f'''* Pandas''')
    st.markdown(f'''* Numpy''')
    st.caption(f'''Install dependencies using [conda](https://docs.conda.io/en/latest/)''')
    st.markdown(f'''<h4 style = 'text-align: left;'> Preprocessing :</h4>''',
                    unsafe_allow_html=True)
    st.markdown(f'''<h7 style = 'text-align: left; opacity: 50%'> _In matches and deliveries dataset have many columns but 
I will be use only few importants columns and will create new columns from existing columns to build the model._
</h7>''',unsafe_allow_html=True)
    st.code('match.info()  # information of match data')
    st.code('''# calculate the sum of first inning and second inning total run for every match
total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()''')
    st.code('''# taking only first innings total_score(means target runs)
total_score_df = total_score_df[total_score_df['inning'] == 1]  ''')
    st.code('total_score_df  # show total_score_df')
    st.code('''# merge both data match and total_score_df on match_id
match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')''')
    st.code('match_df  # show match dataset')
    st.code('''# there have many teams who does not play anymore. so, I will remove those teams from dataset
match_df['team1'].unique()  ''')
    st.code('''# final teams name only 8 ,who are still playing
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]''')
    st.markdown(f'''<h7 style = 'text-align: left; opacity: 100%'> 📝Note:
>Delhi daredevils change the yeam name to Delhi Capitals so, I will change all Delhi Daredevils
>name to Delhi Capitals beacuse it's a same team just changed the name of team,
>and as well as for Sunrises Hydrabad, because previously Sunrises Hyderabad team name was Decan Chargers.
</h7>''',unsafe_allow_html=True)
    st.code('''match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals') 
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')''')
    st.code('''match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')''')
    st.code('''# drop all others teams which does not exist in above team list 
match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]''')
    st.code('''match_df.shape  # shape of match data''')
    st.code('''match_df = match_df[match_df['dl_applied'] == 0]    # dl_applied means the match which does not played 
                                                    # winner decleared by toss or something else.
                                                    # so, I'm taking only those match which played completely''')
    st.code('''match_df = match_df[['match_id','city','winner','total_runs']]  # Add mntion rows to match_df data''')
    st.code('''delivery_df = match_df.merge(delivery,on='match_id')   # merge match data with delivery data''')    
    st.code('''delivery_df = delivery_df[delivery_df['inning'] == 2]    # now only 2nd innings data taking''')
    st.code('delivery_df.head(2)')
    st.code('delivery_df.shape # shape of delivery data')
    st.code('''delivery_df['current_score'] = delivery_df.groupby('match_id').cumsum()['total_runs_y']  ''')
    st.code('delivery_df.head(2) # show delivery data')
    st.code('''# runs left calculate by total_run - current_score
delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score'] ''')
    st.code('delivery_df.head(2) # show delivery data')
    st.code('''# balls left after each ball calculating [formula - 126 - (over*6) + ball]
delivery_df['balls_left'] = 126 - (delivery_df['over']*6 + delivery_df['ball'])''')
    st.code('delivery_df.head(2) # show delivery data')
    st.code('''# calculating wickets 
# all nan values filling with 0 or except nan for any other batsman name set 1 value
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna('0')
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == '0'  else '1')
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int') # change type into 'int'
wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values   # sum wickets after each balls
delivery_df['wickets_left'] = 10 - wickets # calculating wickets left after each ball
delivery_df.head(2) # display delivery data''')
    st.code('''# calculating current run rate
# crr = runs/over
delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])''')
    st.code('delivery_df.head(2) # display delivery data')
    st.code('''# calculating required run rate
# rrr = (runs_left*6) / balls_left
delivery_df['rrr'] =  (delivery_df['runs_left']*6) / delivery_df['balls_left']''')
    st.code('delivery_df.head(2) # show delivery data')
    st.code('''# function for decleard winner
# this function work like if batting teams == winners return 1 or else return 0
def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0''')
    st.code('''delivery_df['result'] = delivery_df.apply(result,axis=1) # apply the result function on delivery data''')
    st.code('delivery_df.head(2) # show delivery data')
    st.code('''# making final data with all important columns
final_data = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left', 'total_runs_x', 'crr','rrr','result']]''')
    st.code('''# shuffle the data 
final_df = final_data.sample(final_data.shape[0])''')
    st.code('''final_df.sample() # show one sample shuffle data''')
    st.code('final_df.isnull().sum() # check missing values')
    st.code('''final_df.dropna(inplace=True) # drop all missing values''')
    st.markdown(f'''<h7 style = 'text-align: left; opacity: 100%'> ❗Problem :

>In final dataset have some problem, last match of every balls the balls_left column become 
>0 so when it devide by  runs_left column for calculating required run rate it's  showing 
>infinity beacuse it's mean 0 devided by some value so, it's become infinity. To fix this problem 
>I dropped all matches last ball where balls_left == 0 dropped that rows.
</h7>''',unsafe_allow_html=True)
    
    st.code('''final_df = final_df[final_df['balls_left'] != 0] # drop all balls_left == 0 roes''')
    st.code('''# separate the data into features(x) and labels(y) 
x = final_df.iloc[:,:-1]  
y = final_df.iloc[:,-1] 

# import train test split
from sklearn.model_selection import train_test_split
# split the data unto train and test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=42)''')
    st.write('Preprocessing is done now for fit the data in algorithm check model build page.')
    st.write('\n')
    st.caption('''👨‍💻For full code with output check my [GitHub](https://github.com/Rafikul10/IPL-win-Probability-Predictor) repositories.''')   
    
    
    
# Model build page --
if val == 2:
    st.write('---')
    st.markdown("<h2 style='text-align: center; color: #FF4B4B;;'> Model Build</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.33,1,0.1])
    with col1:
        pass
    with col2:
        st.write('''[![Scikitlearn](https://img.shields.io/badge/Scikit--learn-1.0.2-orange)](https://scikit-learn.org/stable/tutorial/index.html)
                 [![Logisticregression](https://img.shields.io/badge/Algorithm-LogisticRegression-blueviolet)](https://www.ibm.com/topics/logistic-regression)
                 [![Python 3.10.0](https://img.shields.io/badge/Python-3.10.0-brightgreen)](https://www.python.org/downloads/release/python-3100/) ''')
    with col3:
        pass
    st.write('\n')
    st.markdown(f'''<h4 style = 'text-align: left;'> Dependencies :</h4>''',
                    unsafe_allow_html=True)
    st.markdown(f'''* Jupyter Notebook''') 
    st.markdown(f'''* Python 3.10.0''')
    st.markdown(f'''* scikit-lean''')
    st.markdown(f'''* LogisticRegression''')
    st.markdown(f'''* RandomForestClassifier''')
    
    st.caption(f'''Install dependencies using [conda](https://docs.conda.io/en/latest/)''')
    
    st.markdown(f'''<h4 style = 'text-align: left;'> Model Build :</h4>''',
                    unsafe_allow_html=True)
    
    
    # st.write('**Model Build :**')     
    st.write('\n')
    st.markdown(f'''<h4 style = 'text-align: center; color: #FF4B4B'> 🔡OneHotEncoding 
                </h4>''',unsafe_allow_html=True)
    st.markdown(f'''<h7 style = 'text-align: center; opacity: 50%;'> OneHotEncoding is 
                one method of converting data to prepare it for an algorithm and get a 
                better prediction. With one-hot, I converted each categorical value into 
                a new categorical column and assign a binary value of 1 or 0 to those columns. 
                Each integer value is represented as a binary vector.
                </h7>''',unsafe_allow_html=True)
    st.code('''# import column transformer and one hot encoding 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transform = ColumnTransformer([
    ('str', OneHotEncoder(sparse=False, drop='first'),['batting_team', 'bowling_team', 'city'])
]
,remainder='passthrough')''')
    st.write('\n')
    st.markdown(f'''<h4 style = 'text-align: center; color: #FF4B4B;'> 🎢LogisticRegression
                </h4>''',unsafe_allow_html=True)
    
    st.markdown(f'''<h7 style = 'text-align: left; opacity: 100%;'>

>Logistic regression is a classification algorithm. It is used to predict a binary outcome based on a set of independent variables.

> A binary outcome is one where there are only two possible scenarios—either the event happens (1) or it does not happen (0). Independent variables are those variables or factors which may influence the outcome (or dependent variable).</h7>''',unsafe_allow_html=True)
    st.image('https://i.ibb.co/txV6Q5n/logistic-regression-curve-l.jpg')

    st.markdown(f'''<h7 style = 'text-align: left; opacity: 100%;'>Note : If you want to know more about LogisticRegression click [here](https://www.ibm.com/topics/logistic-regression)
                </h7>''',unsafe_allow_html=True)
    st.write('\n')
    st.code('''from sklearn.linear_model import LogisticRegression  # import LogisticRegression
from sklearn.pipeline import Pipeline                # import Pipeline
from sklearn.ensemble import RandomForestClassifier  # import RandomForestClassifier''')
    st.code('''# create a pipeline of one hot encodeing and LogisticRegression model
pipe = Pipeline(steps=[
    ('step1',transform),
    ('step2',LogisticRegression(solver='liblinear'))
])''')
    st.code('pipe.fit(x_train,y_train) # fit the train data into model')
    st.code('y_pred = pipe.predict(x_test) # predict on test data')
    st.code('''from sklearn.metrics import accuracy_score # import accuracy
accuracy = accuracy_score(y_test,y_pred) # check accuracy on test data
print('Accuracy of the LogisticRegression model is - ',accuracy*100)''')
    st.code('''# create another pirpline for RandomForestClassifier
pipe2 = Pipeline(steps=[
    ('step1',transform),
    ('step2',RandomForestClassifier())
])''')
    st.code('pipe2.fit(x_train,y_train) # Fit the data into RandomForestClassifier model')
    st.code('y_pred_rfc = pipe2.predict(x_test) # predict on test data using RandomForestClassifier model')
    st.code('''accuracy_rfc = accuracy_score(y_test,y_pred_rfc) 
print('Accyracy of the RandomForestClassifier mdoel is - ',accuracy_rfc*100)  # accuracy check on test data''')
    st.code('teams # print teams name used in mdoel')
    st.code('''delivery_df['city'].unique() # print city name used in model''')
    st.write('\n')
    st.markdown(f'''<h7 style = 'text-align: left; color: #ffff;'> 📄Save Model
>By using pickle library saved the model in .pkl format so, i can load this .pkl file again in vs code to use it in Web Application. </h7>''',unsafe_allow_html=True)
    st.code('''import pickle  # import pickle
pickle.dump(pipe,open('LR_pipe.pkl', 'wb'))  # LogisticRegression model save
pickle.dump(pipe2,open('RFC_pipe.pkl', 'wb')) # RandomForestClassifier mdoel save''')

if val == 3:
    st.write('----')
    st.markdown("<h2 style='text-align: center; color: #FF4B4B;;'> Web Application Build</h2>", unsafe_allow_html=True)
    # columns create for align the text in middle
    col1, col2, col3 = st.columns([0.3001,1,0.1])
    with col1:
        pass
    with col2:
        st.write('''[![Python](https://img.shields.io/badge/Python-3.10.0-brightgreen)](https://www.python.org/downloads/release/python-3100/)
                 [![Streamlit](https://img.shields.io/badge/Streamlit%20-1.14.0-Ff0000)](https://docs.streamlit.io/library/get-started)
                 [![MYSQL](https://img.shields.io/badge/DataBase-MySQL-blueviolet)](https://dev.mysql.com/doc/)
                 [![github](https://camo.githubusercontent.com/3a41f9e3f8001983f287f5447462446e6dc1bac996fedafa9ac5dae629c2474f/68747470733a2f2f62616467656e2e6e65742f62616467652f69636f6e2f4769744875623f69636f6e3d67697468756226636f6c6f723d626c61636b266c6162656c)](https://github.com/Rafikul10/IPL-win-Probability-Predictor)''')
    st.write('\n')
    st.markdown(f'''<h4 style = 'text-align: left;'> Dependencies :</h4>''',
                    unsafe_allow_html=True)
    st.markdown(f'''* Python 3.10.0''')
    st.markdown(f'''* Streamlit 1.14.0''')
    st.markdown(f'''* Pandas''')
    st.markdown(f'''* Numpy''')
    st.markdown(f'''* MySQL''')
    st.caption('''This website Application build by using Python, steamlit library and to store the data of all users i used MySQL.
It's an secured🔐 safe Application.''')
    
    st.write('')
    st.subheader('Pages :')
    st.caption('''
               
               >web Application build with total 4page's which are shown below for individual page code is diiferent
               >. If you wanna check how I build
               >the full website and used the machine learning model(.pkl) file in it.
               >Check my [GitHub](https://github.com/Rafikul10/IPL-win-Probability-Predictor) Repositories. Thank You!''')
    st.markdown(f'''* Welcome page''')
    st.markdown(f'''* Home page''') 
    st.markdown(f'''* Project Workflow page''') 
    st.markdown(f'''* Help & Support page''') 
    st.write('---')
    st.write('\n')
    st.subheader('📬 Contact Details')
    st.caption('_If you need any help related website build or ML model build feel free to contact me!!!_')
    st.write(f'📧 ' ,' rafikul.official10@gmail.com',type='mail')
    st.write("👾  [GitHub](https://github.com/Rafikul10?tab=repositories)")
    
if val == 4:
    st.write('---')    
    st.markdown("<h2 style='text-align: center; color: #FF4B4B;;'> Deployment</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([0.39,1,0.1])
    with col1:
        pass
    with col2:
        st.write('''[![Cloud](https://img.shields.io/badge/CloudPlatform-Heroku-blue)](https://devcenter.heroku.com/categories/reference)
                 [![Streamlit](https://img.shields.io/badge/Streamlit%20-1.14.0-Ff0000)](https://streamlit.io/)
                 [![Github](https://camo.githubusercontent.com/3a41f9e3f8001983f287f5447462446e6dc1bac996fedafa9ac5dae629c2474f/68747470733a2f2f62616467656e2e6e65742f62616467652f69636f6e2f4769744875623f69636f6e3d67697468756226636f6c6f723d626c61636b266c6162656c)](https://github.com/Rafikul10/IPL-win-Probability-Predictor)
                 ''')
    with col3:
        pass  
        
    st.caption('''Now Web Application is ready for deploy in any cloud server. In my case i choose Heroku cloud server for deploy the Application. In
               order to deploy the Application on Heroku first create an account and login to your account. After sucessfully logged in.
               You can deploy your application directly by downloading the CLI from official website of
               Heroku for all processs check [Heroku](https://devcenter.heroku.com/categories/reference) official website.''')
    
    st.caption('''for full codeNOTE : For code check my [GitHub](https://github.com/Rafikul10/IPL-win-Probability-Predictor) repositories.''')   
    