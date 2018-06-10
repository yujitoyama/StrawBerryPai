from flask import Flask, render_template, request,session
from pymongo import MongoClient
import pandas as pd
import datetime
import json
import subprocess


app = Flask(__name__)
app.secret_key = 'session'

@app.route('/')
def hello():
    name = 'yuji'
    return render_template('top.html', title='TOP画面', name=name)

@app.route('/graph', methods=['POST'])
def graphview():
    #top画面で選択したmeigara種別を選択
    meigaralabel = request.form['meigara']
    #セッションに選択したrisk種別を保存
    session['meigara'] = meigaralabel
    #mongoDBへ接続    
    client = MongoClient('localhost', 27017)
    db = client.swdatadb

    #tes1 collection(NTTYY)データのdataframe化
    pipe = [{'$project':{'_id': 0, 'Date':1 ,'Close': 1}}]    
    agg = db.tes1.aggregate(pipeline = pipe)
    
    Data1 = []

    '''
    Datevlues = []
    Closevalues = []
    for r in agg:
        Datevlues.append(r['Date'])
        Closevalues.append(r['Close'])
    Data1 = {'Date':Datevlues,'Close':Closevalues}
    '''

    for r in agg:
        Data1.append(r)

    #辞書→Dataframeへの変換
    Data1_df = pd.DataFrame.from_dict(Data1)
    #Dataframe→JSONへの変換
    Data1_json = Data1_df.to_json()

    return render_template('gview.html', title='グラフ画面',graphvalues = Data1)

@app.route('/soukan', methods=['POST'])
def soukanview():
    #mongoDBへ接続    
    client = MongoClient('localhost', 27017)
    db = client.swdatadb

    #tes1 collection(NTTYY)データのdataframe化
    pipe = [{'$project':{'_id': 0 ,'Close': 1}}]    
    agg1 = db.tes1.aggregate(pipeline = pipe)
    agg2 = db.tes2.aggregate(pipeline = pipe)
    agg3 = db.tes3cht.aggregate(pipeline = pipe)
    agg4 = db.tes4cha.aggregate(pipeline = pipe)

    Data1 = []
    Data2 = []
    Data3 = []
    Data4 = []

    Closevalues1 = []
    for r in agg1:
        Closevalues1.append(r['Close'])
    Data1 = {'Close':Closevalues1}

    Closevalues2 = []
    for r in agg2:
        Closevalues2.append(r['Close'])
    Data2 = {'Close':Closevalues2}

    Closevalues3 = []
    for r in agg3:
        Closevalues3.append(r['Close'])
    Data3 = {'Close':Closevalues3}

    Closevalues4 = []
    for r in agg4:
        Closevalues4.append(r['Close'])
    Data4 = {'Close':Closevalues4}

    #辞書→Dataframeへの変換
    Data1_df = pd.DataFrame.from_dict(Data1)
    #辞書→Dataframeへの変換
    Data2_df = pd.DataFrame.from_dict(Data2)
    #辞書→Dataframeへの変換
    Data3_df = pd.DataFrame.from_dict(Data3)
    #辞書→Dataframeへの変換
    Data4_df = pd.DataFrame.from_dict(Data4)

    #相関係数の判定ロジック
    tmp = pd.DataFrame()
    tmp['NTTD'] = Data1_df['Close']
    tmp['KDDI'] = Data2_df['Close']
    tmp['CHA'] = Data3_df['Close']
    tmp['CHT'] = Data4_df['Close']
    #r = tmp.corr()
    r = tmp.corr().iloc[:,0]
    '''
    rj = r.to_json()
    print(type(r))
    print(r)
    print(type(rj))
    print(rj)
    '''

    trydata = []
    indexs = ["NTTD","KDDI","CHA","CHT"]
    for index in indexs:
        tmp = {"meigara":index,"values":r[index]}
        trydata.append(tmp)

    '''
    for index in INDEX_LIST[1:]: 
        tmp[index] = closing_data[index]
        r = tmp.corr().iloc[:, 0]
    print(r)
    '''

    return render_template('soukan.html', title='相関係数画面',soukanvalues = trydata)


@app.route('/predict', methods=['POST'])
def predict():

    args = ['python', './analyzefunc/analyze.py']
    try:
        subprocess.check_call(args)
    except:
        print("analyze failed")

    #predictreslts  = pd.read_csv("./analyzefunc/testdata/result.csv")
    return render_template('top.html', title='TOP画面', done="予測完了！")

@app.route('/pview', methods=['POST'])
def pview():

    with open('./analyzefunc/testdata/' + 'result.txt', 'r') as f:
        predictreslts = f.read()
    
    if(predictreslts == "[0]"):
        predictreslts = "0"
    elif(predictreslts == "[1]"):
        predictreslts = "1"

    return render_template('pview.html', title='TOP画面', predictreslts=predictreslts)

if __name__ == "__main__":
    app.run(debug=True)