# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:12:20 2020

@author: ladduu
"""

from flask import Flask,render_template,url_for,request,jsonify

# Importing the libraries
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ipl_')
def ipl_():
	return render_template('ipl.html')

@app.route('/ipl', methods=['POST'])

def ipl():
    
    # Loading the dataset
    df = pd.read_csv('ipl.csv')
    
    # Removing unwanted columns
    columns_to_remove = ['mid', 'batsman', 'bowler', 'striker', 'non-striker']
    
    print('Before removing unwanted columns: {}'.format(df.shape))
    df.drop(labels=columns_to_remove, axis=1, inplace=True)
    print('After removing unwanted columns: {}'.format(df.shape))
    
    df.columns
    df['bat_team'].unique()
    
    consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                        'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                        'Delhi Daredevils', 'Sunrisers Hyderabad']
    
    
    # Keeping only consistent teams
    print('Before removing inconsistent teams: {}'.format(df.shape))
    df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
    print('After removing inconsistent teams: {}'.format(df.shape))
    
    df['bat_team'].unique()
    
    # Removing the first 5 overs data in every match
    print('Before removing first 5 overs data: {}'.format(df.shape))
    df = df[df['overs']>=5.0]
    print('After removing first 5 overs data: {}'.format(df.shape))
    
    
    from datetime import datetime
    print("Before converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    print("After converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
    
    
    df.venue.value_counts().sort_values(ascending=False)
    
    top_10 = [x for x in df.venue.value_counts().sort_values(ascending=False).head(10).index]
    
    for label in top_10:
        df[label] = np.where(df['venue']==label,1,0)
        
    
    df.drop(labels='venue', axis=1, inplace=True)    
        
    # Converting categorical features using OneHotEncoding method
    encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
    encoded_df.columns
    
    # Rearranging the columns
    encoded_df = encoded_df[['date','Eden Gardens','M Chinnaswamy Stadium','Feroz Shah Kotla',
                             'Wankhede Stadium','MA Chidambaram Stadium, Chepauk','Punjab Cricket Association Stadium, Mohali',
                             'Sawai Mansingh Stadium','Rajiv Gandhi International Stadium, Uppal','Sardar Patel Stadium, Motera',
                             'Kingsmead', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
                             'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
                             'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
                             'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
                             'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
                             'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
                             'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]
    
    
    # Splitting the data into train and test set
    X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
    X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]
    
    
    y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
    y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values
    
    # Removing the 'date' column
    X_train.drop(labels='date', axis=True, inplace=True)
    X_test.drop(labels='date', axis=True, inplace=True)
    
    
    
    
    
    # Linear Regression Model
    from sklearn.linear_model import LinearRegression
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train,y_train)
    
    
    # Predicting results
    y_pred_lr = linear_regressor.predict(X_test)
    
    
    # Linear Regression - Model Evaluation
    from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, accuracy_score
    print("---- Linear Regression - Model Evaluation ----")
    print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_lr)))
    print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_lr)))
    print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_lr))))
    
    
    # Creating a pickle file for the regressor
    filename = 'ipl_model.pkl'
    pickle.dump(linear_regressor, open(filename, 'wb'))
    
    # opening a pickle file for the regressor
    filename = open('ipl_model.pkl','rb')
    linear_regressor = pickle.load(filename)
    
    def predict_score(venue = 'Eden Garden' ,batting_team='Chennai Super Kings', bowling_team='Mumbai Indians', overs=5.1, runs=50, wickets=0, runs_in_prev_5=50, wickets_in_prev_5=0):
      temp_array = list()
      
      #venue
      if venue == 'Eden Garden':
          temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0]
      elif venue == 'M Chinnaswamy Stadium':
          temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0]
      elif venue == 'Feroz Shah Kotla':
          temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0]
      elif venue == 'Wankhede Stadium':
          temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0]
      elif venue == 'MA Chidambaram Stadium, Chepauk':
          temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0]
      elif venue == 'Punjab Cricket Association Stadium, Mohali':
          temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0]
      elif venue == 'Sawai Mansingh Stadium':
          temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0]
      elif venue == 'Rajiv Gandhi International Stadium, Uppal':
          temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0]
      elif venue == 'Sardar Patel Stadium, Motera':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0]
      elif venue == 'Kingsmead':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1]
    
    
      # Batting Team
      if batting_team == 'Chennai Super Kings':
          temp_array = temp_array + [1,0,0,0,0,0,0,0]
      elif batting_team == 'Delhi Daredevils':
          temp_array = temp_array + [0,1,0,0,0,0,0,0]
      elif batting_team == 'Kings XI Punjab':
          temp_array = temp_array + [0,0,1,0,0,0,0,0]
      elif batting_team == 'Kolkata Knight Riders':
          temp_array = temp_array + [0,0,0,1,0,0,0,0]
      elif batting_team == 'Mumbai Indians':
          temp_array = temp_array + [0,0,0,0,1,0,0,0]
      elif batting_team == 'Rajasthan Royals':
          temp_array = temp_array + [0,0,0,0,0,1,0,0]
      elif batting_team == 'Royal Challengers Bangalore':
          temp_array = temp_array + [0,0,0,0,0,0,1,0]
      elif batting_team == 'Sunrisers Hyderabad':
          temp_array = temp_array + [0,0,0,0,0,0,0,1]
    
      # Bowling Team
      if bowling_team == 'Chennai Super Kings':
          temp_array = temp_array + [1,0,0,0,0,0,0,0]
      elif bowling_team == 'Delhi Daredevils':
          temp_array = temp_array + [0,1,0,0,0,0,0,0]
      elif bowling_team == 'Kings XI Punjab':
          temp_array = temp_array + [0,0,1,0,0,0,0,0]
      elif bowling_team == 'Kolkata Knight Riders':
          temp_array = temp_array + [0,0,0,1,0,0,0,0]
      elif bowling_team == 'Mumbai Indians':
          temp_array = temp_array + [0,0,0,0,1,0,0,0]
      elif bowling_team == 'Rajasthan Royals':
          temp_array = temp_array + [0,0,0,0,0,1,0,0]
      elif bowling_team == 'Royal Challengers Bangalore':
          temp_array = temp_array + [0,0,0,0,0,0,1,0]
      elif bowling_team == 'Sunrisers Hyderabad':
          temp_array = temp_array + [0,0,0,0,0,0,0,1]
    
      # Overs, Runs, Wickets, Runs_in_prev_5, Wickets_in_prev_5
      temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
    
      # Converting into numpy array
      temp_array = np.array([temp_array])
      # Prediction
      return int(linear_regressor.predict(temp_array)[0])
    #write ML program here
    if request.method == 'POST':
        ven=request.form['ven']
        bat_team=request.form['bt']
        bowl_team=request.form['blt']
        over=float(request.form['over'])
        run=int(request.form['run'])
        wic=int(request.form['wic'])
        rinp=int(request.form['rinp'])
        winp=int(request.form['winp'])
        connect=5
        final_score = predict_score(venue=ven,batting_team=bat_team, bowling_team=bowl_team, overs=over, runs=run,wickets=wic, runs_in_prev_5=rinp, wickets_in_prev_5=winp)
    return render_template('ipl.html',connect=connect,final_score_l=final_score-5,final_score_u=final_score+7)

@app.route('/odi_')
def odi_():
	return render_template('odi.html')

@app.route('/odi', methods=['POST'])
def odi():
    
    # Loading the dataset
    df = pd.read_csv('odi.csv')
    
    # Removing unwanted columns
    columns_to_remove = ['mid', 'batsman', 'bowler', 'striker', 'non-striker']
    
    print('Before removing unwanted columns: {}'.format(df.shape))
    df.drop(labels=columns_to_remove, axis=1, inplace=True)
    print('After removing unwanted columns: {}'.format(df.shape))
    
    df.columns
    df['bat_team'].unique()
    
    
    consistent_teams = ['England', 'Australia', 'South Africa', 'Sri Lanka', 'West Indies',
                        'Kenya', 'Pakistan', 'India', 'New Zealand', 'Bangladesh',
                        'Scotland', 'Ireland', 'Afghanistan', 'Zimbabwe', 'Canada']
    
    # Keeping only consistent teams
    print('Before removing inconsistent teams: {}'.format(df.shape))
    df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
    print('After removing inconsistent teams: {}'.format(df.shape))
    
    # Removing the first 5 overs data in every match
    print('Before removing first 30 overs data: {}'.format(df.shape))
    df = df[df['overs']>=30.0]
    print('After removing first 30 overs data: {}'.format(df.shape))
    
    
    from datetime import datetime
    print("Before converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    print("After converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
    
    df.venue.value_counts().sort_values(ascending=False)
    
    top_10 = [x for x in df.venue.value_counts().sort_values(ascending=False).head(10).index]
    
    
    for label in top_10:
        df[label] = np.where(df['venue']==label,1,0)
    
    df.drop(labels='venue', axis=1, inplace=True)
    
    # Converting categorical features using OneHotEncoding method
    encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
    encoded_df.columns
    
    # Rearranging the columns
    encoded_df = encoded_df[['date','Shere Bangla National Stadium', 'Harare Sports Club',
                             'R Premadasa Stadium', 'Queens Sports Club', 'Sheikh Zayed Stadium',
                             'Melbourne Cricket Ground', 'Sydney Cricket Ground', 'Adelaide Oval',
                             'Rangiri Dambulla International Stadium','Western Australia Cricket Association Ground',
                             'bat_team_Afghanistan', 'bat_team_Australia',
                             'bat_team_Bangladesh','bat_team_Canada', 'bat_team_England', 'bat_team_India',
                             'bat_team_Ireland', 'bat_team_Kenya', 'bat_team_New Zealand',
                             'bat_team_Pakistan', 'bat_team_Scotland', 'bat_team_South Africa',
                             'bat_team_Sri Lanka', 'bat_team_West Indies', 'bat_team_Zimbabwe',
                             'bowl_team_Afghanistan', 'bowl_team_Australia', 'bowl_team_Bangladesh',
                             'bowl_team_Canada', 'bowl_team_England', 'bowl_team_India',
                             'bowl_team_Ireland', 'bowl_team_Kenya', 'bowl_team_New Zealand',
                             'bowl_team_Pakistan', 'bowl_team_Scotland', 'bowl_team_South Africa',
                             'bowl_team_Sri Lanka', 'bowl_team_West Indies', 'bowl_team_Zimbabwe',
                             'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]
    
    
    
    
    # Splitting the data into train and test set
    X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2015]
    X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2016]
    
    
    y_train = encoded_df[encoded_df['date'].dt.year <= 2015]['total'].values
    y_test = encoded_df[encoded_df['date'].dt.year >= 2016]['total'].values
    
    # Removing the 'date' column
    X_train.drop(labels='date', axis=True, inplace=True)
    X_test.drop(labels='date', axis=True, inplace=True)
    
    # Linear Regression Model
    from sklearn.linear_model import LinearRegression
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train,y_train)
    
    # Predicting results
    y_pred_lr = linear_regressor.predict(X_test)
    
    # Linear Regression - Model Evaluation
    from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, accuracy_score
    print("---- Linear Regression - Model Evaluation ----")
    print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_lr)))
    print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_lr)))
    print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_lr))))
    
    # Creating a pickle file for the regressor
    filename = 'odi_model.pkl'
    pickle.dump(linear_regressor, open(filename, 'wb'))
    
    # opening a pickle file for the regressor
    filename = open('odi_model.pkl','rb')
    linear_regressor = pickle.load(filename)
    
    
    def predict_score(venue = 'Shere Bangla National Stadium' ,batting_team='India', bowling_team='Pakistan', overs=32.1, runs=200, wickets=2, runs_in_prev_5=58, wickets_in_prev_5=0):
      temp_array = list()
      
      #venue
      if venue == 'Shere Bangla National Stadium':
          temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0]
      elif venue == 'Harare Sports Club':
          temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0]
      elif venue == 'R Premadasa Stadium':
          temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0]
      elif venue == 'Queens Sports Club':
          temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0]
      elif venue == 'Sheikh Zayed Stadium':
          temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0]
      elif venue == 'Melbourne Cricket Ground':
          temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0]
      elif venue == 'Sydney Cricket Ground':
          temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0]
      elif venue == 'Adelaide Oval':
          temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0]
      elif venue == 'Rangiri Dambulla International Stadium':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0]
      elif venue == 'Western Australia Cricket Association Ground':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1]
    
    
      # Batting Team
      if batting_team == 'Afghanistan':
          temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
      elif batting_team == 'Australia':
          temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
      elif batting_team == 'Bangladesh':
          temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
      elif batting_team == 'Canada':
          temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
      elif batting_team == 'England':
          temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
      elif batting_team == 'India':
          temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
      elif batting_team == 'Ireland':
          temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
      elif batting_team == 'Kenya':
          temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
      elif batting_team == 'New Zealand':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
      elif batting_team == 'Pakistan':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
      elif batting_team == 'Scotland':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
      elif batting_team == 'South Africa':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
      elif batting_team == 'Sri Lanka':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
      elif batting_team == 'West Indies':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
      elif batting_team == 'Zimbabwe':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    
      # Bowling Team
      if bowling_team == 'Afghanistan':
          temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
      elif bowling_team == 'Australia':
          temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
      elif bowling_team == 'Bangladesh':
          temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
      elif bowling_team == 'Canada':
          temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
      elif bowling_team == 'England':
          temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
      elif bowling_team == 'India':
          temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
      elif bowling_team == 'Ireland':
          temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
      elif bowling_team == 'Kenya':
          temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
      elif bowling_team == 'New Zealand':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
      elif bowling_team == 'Pakistan':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
      elif bowling_team == 'Scotland':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
      elif bowling_team == 'South Africa':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
      elif bowling_team == 'Sri Lanka':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
      elif bowling_team == 'West Indies':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
      elif bowling_team == 'Zimbabwe':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    
      # Overs, Runs, Wickets, Runs_in_prev_5, Wickets_in_prev_5
      temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
    
      # Converting into numpy array
      temp_array = np.array([temp_array])
    
      # Prediction
      return int(linear_regressor.predict(temp_array)[0])
    #write ML program here
    if request.method == 'POST':
        ven=request.form['ven']
        bat_team=request.form['bt']
        bowl_team=request.form['blt']
        over=float(request.form['over'])
        run=int(request.form['run'])
        wic=int(request.form['wic'])
        rinp=int(request.form['rinp'])
        winp=int(request.form['winp'])
        connect=5
        final_score = predict_score(venue=ven,batting_team=bat_team, bowling_team=bowl_team, overs=over, runs=run,wickets=wic, runs_in_prev_5=rinp, wickets_in_prev_5=winp)
    return render_template('odi.html',connect=connect,final_score_l=final_score-7,final_score_u=final_score+7)

@app.route('/tt_')
def tt_():
	return render_template('tt.html')

@app.route('/tt', methods=['POST'])
def tt():
    
    #write ML program here
        # Loading the dataset
    df = pd.read_csv('t20.csv')
    
    # Removing unwanted columns
    columns_to_remove = ['mid', 'batsman', 'bowler', 'striker', 'non-striker']
    
    print('Before removing unwanted columns: {}'.format(df.shape))
    df.drop(labels=columns_to_remove, axis=1, inplace=True)
    print('After removing unwanted columns: {}'.format(df.shape))
    
    df.columns
    df['bat_team'].unique()
    
    consistent_teams = ['England', 'Australia', 'South Africa', 'Sri Lanka', 'West Indies',
                        'Kenya', 'Pakistan', 'India', 'New Zealand', 'Bangladesh',
                        'Scotland', 'Ireland', 'Afghanistan', 'Zimbabwe', 'Canada']
    
    
    # Keeping only consistent teams
    print('Before removing inconsistent teams: {}'.format(df.shape))
    df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
    print('After removing inconsistent teams: {}'.format(df.shape))
    
    # Removing the first 5 overs data in every match
    print('Before removing first 5 overs data: {}'.format(df.shape))
    df = df[df['overs']>=5.0]
    print('After removing first 5 overs data: {}'.format(df.shape))
    
    
    from datetime import datetime
    print("Before converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    print("After converting 'date' column from string to datetime object: {}".format(type(df.iloc[0,0])))
    
    
    df.venue.value_counts().sort_values(ascending=False)
    
    top_10 = [x for x in df.venue.value_counts().sort_values(ascending=False).head(10).index]
    
    for label in top_10:
        df[label] = np.where(df['venue']==label,1,0)
        
    df.drop(labels='venue', axis=1, inplace=True)
    
    # Converting categorical features using OneHotEncoding method
    encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
    encoded_df.columns
    
    # Rearranging the columns
    encoded_df = encoded_df[['date','Dubai International Cricket Stadium','Shere Bangla National Stadium',
                             'R Premadasa Stadium','New Wanderers Stadium', 'Kensington Oval, Bridgetown',
                             'Pallekele International Cricket Stadium', 'Newlands','Harare Sports Club',
                             'Kennington Oval', 'Eden Park','bat_team_Afghanistan', 'bat_team_Australia',
                             'bat_team_Bangladesh','bat_team_Canada', 'bat_team_England', 'bat_team_India',
                             'bat_team_Ireland', 'bat_team_Kenya', 'bat_team_New Zealand',
                             'bat_team_Pakistan', 'bat_team_Scotland', 'bat_team_South Africa',
                             'bat_team_Sri Lanka', 'bat_team_West Indies', 'bat_team_Zimbabwe',
                             'bowl_team_Afghanistan', 'bowl_team_Australia', 'bowl_team_Bangladesh',
                             'bowl_team_Canada', 'bowl_team_England', 'bowl_team_India',
                             'bowl_team_Ireland', 'bowl_team_Kenya', 'bowl_team_New Zealand',
                             'bowl_team_Pakistan', 'bowl_team_Scotland', 'bowl_team_South Africa',
                             'bowl_team_Sri Lanka', 'bowl_team_West Indies', 'bowl_team_Zimbabwe',
                             'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]
    
    
    # Splitting the data into train and test set
    X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2015]
    X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2016]
    
    
    y_train = encoded_df[encoded_df['date'].dt.year <= 2015]['total'].values
    y_test = encoded_df[encoded_df['date'].dt.year >= 2016]['total'].values
    
    # Removing the 'date' column
    X_train.drop(labels='date', axis=True, inplace=True)
    X_test.drop(labels='date', axis=True, inplace=True)
    
    # Linear Regression Model
    from sklearn.linear_model import LinearRegression
    linear_regressor = LinearRegression()
    linear_regressor.fit(X_train,y_train)
    
    # Predicting results
    y_pred_lr = linear_regressor.predict(X_test)
    
    # Linear Regression - Model Evaluation
    from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, accuracy_score
    print("---- Linear Regression - Model Evaluation ----")
    print("Mean Absolute Error (MAE): {}".format(mae(y_test, y_pred_lr)))
    print("Mean Squared Error (MSE): {}".format(mse(y_test, y_pred_lr)))
    print("Root Mean Squared Error (RMSE): {}".format(np.sqrt(mse(y_test, y_pred_lr))))
    
    # Creating a pickle file for the regressor
    filename = 't20_model.pkl'
    pickle.dump(linear_regressor, open(filename, 'wb'))
    
    # opening a pickle file for the regressor
    filename = open('t20_model.pkl','rb')
    linear_regressor = pickle.load(filename)
    
    
    def predict_score(venue = 'Dubai International Cricket Stadium' ,batting_team='India', bowling_team='Pakistan', overs=5.1, runs=60, wickets=0, runs_in_prev_5=58, wickets_in_prev_5=0):
      temp_array = list()
      
      #venue
      if venue == 'Dubai International Cricket Stadium':
          temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0]
      elif venue == 'Shere Bangla National Stadium':
          temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0]
      elif venue == 'R Premadasa Stadium':
          temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0]
      elif venue == 'New Wanderers Stadium':
          temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0]
      elif venue == 'Kensington Oval, Bridgetown':
          temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0]
      elif venue == 'Pallekele International Cricket Stadium':
          temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0]
      elif venue == 'Newlands':
          temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0]
      elif venue == 'Harare Sports Club':
          temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0]
      elif venue == 'Kennington Oval':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0]
      elif venue == 'Eden Park':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1]
    
    
      # Batting Team
      if batting_team == 'Afghanistan':
          temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
      elif batting_team == 'Australia':
          temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
      elif batting_team == 'Bangladesh':
          temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
      elif batting_team == 'Canada':
          temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
      elif batting_team == 'England':
          temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
      elif batting_team == 'India':
          temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
      elif batting_team == 'Ireland':
          temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
      elif batting_team == 'Kenya':
          temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
      elif batting_team == 'New Zealand':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
      elif batting_team == 'Pakistan':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
      elif batting_team == 'Scotland':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
      elif batting_team == 'South Africa':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
      elif batting_team == 'Sri Lanka':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
      elif batting_team == 'West Indies':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
      elif batting_team == 'Zimbabwe':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    
      # Bowling Team
      if bowling_team == 'Afghanistan':
          temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
      elif bowling_team == 'Australia':
          temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
      elif bowling_team == 'Bangladesh':
          temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
      elif bowling_team == 'Canada':
          temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
      elif bowling_team == 'England':
          temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
      elif bowling_team == 'India':
          temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
      elif bowling_team == 'Ireland':
          temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
      elif bowling_team == 'Kenya':
          temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
      elif bowling_team == 'New Zealand':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
      elif bowling_team == 'Pakistan':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
      elif bowling_team == 'Scotland':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
      elif bowling_team == 'South Africa':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
      elif bowling_team == 'Sri Lanka':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
      elif bowling_team == 'West Indies':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
      elif bowling_team == 'Zimbabwe':
          temp_array = temp_array + [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    
      # Overs, Runs, Wickets, Runs_in_prev_5, Wickets_in_prev_5
      temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
    
      # Converting into numpy array
      temp_array = np.array([temp_array])
    
      # Prediction
      return int(linear_regressor.predict(temp_array)[0])
    #write ML program here
    if request.method == 'POST':
        ven=request.form['ven']
        bat_team=request.form['bt']
        bowl_team=request.form['blt']
        over=float(request.form['over'])
        run=int(request.form['run'])
        wic=int(request.form['wic'])
        rinp=int(request.form['rinp'])
        winp=int(request.form['winp'])
        connect=5
        final_score = predict_score(venue=ven,batting_team=bat_team, bowling_team=bowl_team, overs=over, runs=run,wickets=wic, runs_in_prev_5=rinp, wickets_in_prev_5=winp)
    return render_template('tt.html',connect=connect,final_score_l=final_score-7,final_score_u=final_score+7)

if __name__ == '__main__':
    app.run(debug=True)


