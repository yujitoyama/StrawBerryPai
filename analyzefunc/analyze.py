from pandas_datareader.data import DataReader
from datetime import datetime
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import subprocess
import csv

INDEX_LIST = ["NTTYY",   #NTT
            "T",  #ATT
            "TI",   #Telecom Italia S.p.A.
            "TEO",  #Telecom Argentina S.A. 
            "Z74.SI",   #Singapore Telecommunications Limited
            "CHA", #China Telecom Corporation Limited
            "VOX", #Vanguard Telecommunication Services ETF 
            "CMTL", #Comtech Telecommunications Corp.
            "SKM", #SK Telecom Co., Ltd.
            "CHT", #Chunghwa Telecom Co., Ltd. 
            "KDDIY"] #KDDI Corporation

class StockData(object):
    def __init__(self):
        '''
        testdata配置用ディレクトリ作成コンストラクタ
        '''
        self.basedir = "./analyzefunc/testdata/"
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)        

    def get_closing_data(self):
        '''
        csvファイルを読み込んで、各ファイルの終値を抽出、正規化を行う関数
        '''
        predata = pd.read_csv(self.basedir + "data.csv")
        '''
        #正規化
        for index in INDEX_LIST:
            if normalize:
                closing_data[index] = closing_data[index] / max(closing_data[index])
            if logreturn:
                closing_data[index] = np.log(closing_data[index] / closing_data[index].shift())
        return closing_data
        '''
        return predata

    def inference(self, feature_data, layers):
        '''
        ディープラーニングモデル
        '''
        if len(layers) < 2:
            raise Exception("'layers' should have more than one elements")
        previous_layer = feature_data
        for i in range(len(layers) - 2):
            with tf.name_scope("Hidden" + str(i + 1)):
                weights = tf.Variable(tf.truncated_normal([layers[i], layers[i + 1]], stddev=0.0001))
                biases = tf.Variable(tf.ones([layers[i + 1]]))
                previous_layer = tf.nn.relu(tf.matmul(previous_layer, weights) + biases)
        with tf.name_scope("Output"):
            weights = tf.Variable(tf.truncated_normal([layers[-2], layers[-1]], stddev=0.0001))
            biases = tf.Variable(tf.ones([layers[-1]]))
            model = tf.nn.softmax(tf.matmul(previous_layer, weights) + biases)
        return model

if __name__ == '__main__':
    s = StockData()
    predata = s.get_closing_data()

    '''
    #closing_dataにcolumn=N225_positiveを作成し、初期値として全てに0を設定
    closing_data["NTTYY_positive"] = 0
    #NTTが>0となる行について、AHKSY_positive列を1に変更
    closing_data.ix[closing_data["NTTYY"] >= 0, "NTTYY_positive"] = 1
    closing_data["NTTYY_negative"] = 0
    closing_data.ix[closing_data["NTTYY"] < 0, "NTTYY_negative"] = 1

    training_test_data = pd.DataFrame(
    # column name is "<index>_<day>".
    # E.g., "DJI_1" means yesterday's Dow.
    columns= ["NTTYY_positive", "NTTYY_negative"] + [s + "_1" for s in INDEX_LIST[1:]]
    )

    for i in range(7, len(closing_data)):
        data = {}
        # We will use today's data for positive/negative labels
        data["NTTYY_positive"] = closing_data["NTTYY_positive"].ix[i]
        data["NTTYY_negative"] = closing_data["NTTYY_negative"].ix[i]
        # Use yesterday's data for world market data
        for col in INDEX_LIST[1:]:
            data[col + "_1"] = closing_data[col].ix[i]
        training_test_data = training_test_data.append(data, ignore_index=True)

    #学習データ0.8,テストデータ0.2でデータを分ける
    #モデルへの入力値を3列目以降、正解を2列目までとした
    predictors_tf = training_test_data[training_test_data.columns[2:]]
    classes_tf = training_test_data[training_test_data.columns[:2]]
    training_set_size = int(len(training_test_data) * 0.8)
    test_set_size = len(training_test_data) - training_set_size
    training_predictors_tf = predictors_tf[:training_set_size]
    training_classes_tf = classes_tf[:training_set_size]
    test_predictors_tf = predictors_tf[training_set_size:]
    test_classes_tf = classes_tf[training_set_size:]


    '''
    predictor_tf = predata
    num_predictors = len(predictor_tf.columns)
    ###num_classes = len(training_classes_tf.columns)

    ###feature_data = tf.placeholder("float", [None, num_predictors])
    ##actual_classes = tf.placeholder("float", [None, num_classes])

    #隠れ層情報をパラメータ化
    ###layers = [num_predictors,30,num_classes]

    #model = s.inference(feature_data, num_predictors)

    ##cost = s.loss(model, actual_classes)
    ##tf.summary.scalar('cost',cost)
    ##training_step = s.training(cost,global_step, learning_rate=0.0001)
    ##correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
    ##accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    feature_data = tf.placeholder("float", [None, num_predictors])
    layers = [10,30,2]
    ##global_step = tf.Variable(0, name='global_step', trainable=False)
    model = s.inference(feature_data,layers)

    #初期化
    sess = tf.Session()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        ckpt_state = tf.train.get_checkpoint_state('./analyzefunc/models/')

        if ckpt_state:
            last_model = ckpt_state.model_checkpoint_path
            saver.restore(sess,last_model)
            print("model was loaded:", last_model)
        else:
            sess.run(init)
            print("initialized.")

        ans = sess.run(
            tf.argmax(model,1),
            feed_dict={
                feature_data: predictor_tf.values
                    #actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
            }
        )

        with open('./analyzefunc/testdata/' + 'result.txt', 'w') as f:
            f.write(str(ans))

        with open('./analyzefunc/testdata/' + 'result.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
            writer.writerow(ans) 