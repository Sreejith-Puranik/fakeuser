from flask import Flask, render_template, request, jsonify, session
from retrieve_tweet import data_collection,download_user,download_user_bulk
import joblib
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.color'] ='tab:orange'
from analisis_data_profil import preprocess, preprocess_bulk
app = Flask(__name__)
app.secret_key = 'twitter'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

#SESSION_TYPE = 'filesystem'

model = joblib.load('finalized_model_without.sav')
path = os.getcwd()+'\\collectindividual'


def prediction_bulk(df):
    skrip = preprocess_bulk(df)
    pred_proba=model.predict_proba(skrip)
    y_pred=model.predict(skrip)
    percentage=pred_proba[:,1]
    joins=' '.join(map(str, percentage))
    perc=float(joins)*100
    percent=(str(perc)+"%")
    #print(y_pred)
    df = df.drop(df.columns[[17,18,19,20,21,22]], axis = 1)
    return df,percent,y_pred

def prediction(df):
    skrip,tab = preprocess(df)
    pred_proba=model.predict_proba(skrip)
    y_pred=model.predict(skrip)
    percentage=pred_proba[:,1]
    joins=' '.join(map(str, percentage))
    perc=float(joins)*100
    percent=perc
    print(y_pred)
    return percent,tab,y_pred

@app.route("/")
def login():
    return render_template('login.html')

@app.route("/collect")
def collect():
    print("------------------------------------------------")
    df = data_collection()
    print(df)
    print("------------------------------------------------")
    df=df.sort_values(by=['username']).reset_index(drop=True)
    df.to_csv('collect.csv') 
    return render_template('collect.html',df=df.to_html())

@app.route("/test")
def test():
    df = pd.read_csv("collect.csv")
    res = pd.DataFrame()
    for ind in df.index:
        uname = df['username'][ind]
        print(uname)
        download_user_bulk(df['username'][ind])
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
        fr,per,stat=prediction_bulk(df)
        pr = per
        st = stat
        fr['Percentage'] = pr
        if st == False :
            Result = "Legitimate User"
            fr['State'] = Result
        else:
            Result = "Fake/Bot User"
            fr['State'] = Result
        res = res.append(fr)
    res=res.reset_index(drop=True)
    return render_template('test.html',result=res.to_html())

@app.route("/check", methods=['POST','GET'])
def check():
    dl_user = request.form.get('chat_in')
    if dl_user == None:
         return render_template('check.html')
    else:
        session['username'] = dl_user     
        download = download_user(dl_user)
        df=pd.read_csv('coba.csv')
        return render_template('check.html',df=df.to_html())

@app.route("/detect")
def detect():
    all_files = glob.glob(path + "/*.csv")
    for filename in all_files:
        if filename.endswith('.csv'):
            os.unlink(filename)
            #print(filename)
    my_var = session.get('username', None)
    print(my_var)
    df=pd.read_csv('coba.csv')
    prediksi,tab,y_pred=prediction(df)
    tab['Username']=my_var
    tab=tab.set_index('Username')
    tab=tab
    labels = 'Legitimate', 'Fake'
    size1 = 100-float(prediksi)
    size2 = prediksi
    sizes = [size1, size2]
    colors = ['forestgreen', 'crimson']
    explode = (0, 0.15)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', colors =colors,
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('static//images//graph//'+my_var+'.png',transparent=True)
    img = 'static/images/graph/'+my_var+'.png'
    return render_template('detect.html',prediction=prediksi,tab=tab.to_html(),y_pred=y_pred,my_var=my_var,img=img)
   
if __name__ == "__main__":
    app.run(debug=True)   

