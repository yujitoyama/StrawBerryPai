from pandas_datareader.data import DataReader
from datetime import datetime
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import subprocess

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
        data、log配置用ディレクトリ作成コンストラクタ
        '''
        self.basedir = "./data/"
        self.LOG_DIR = os.path.join(os.path.dirname(__file__),'log')
        self.efdatadir = "./efdata/"
        self.models = "./models/"
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)
        if not os.path.exists(self.LOG_DIR):
            os.mkdir(self.LOG_DIR)
        if not os.path.exists(self.efdatadir):
            os.mkdir(self.efdatadir)   
        if not os.path.exists(self.models):
            os.mkdir(self.models)         

    def getcsv(self):
        '''
        pandas DataReaderを利用してYahooFinanceから過去一年ぶんの株価データを抽出、
        csvファイル化する関数
        '''
        if os.path.exists(self.basedir):
            end = datetime.now()
            start = datetime(end.year - 1, end.month, end.day)
            for index in INDEX_LIST:
                filename = self.basedir + index + ".csv"
                price = DataReader(index, 'yahoo', start, end)
                price.to_csv(filename)

    def get_closing_data(self, days=100, normalize=False, logreturn=False):
        '''
        csvファイルを読み込んで、各ファイルの終値を抽出、正規化を行う関数
        '''
        closing_data = pd.DataFrame()
        for index in INDEX_LIST:
            df = pd.read_csv(self.basedir + index + ".csv").set_index('Date')
            closing_data[index] = df["Close"][:days] if days else df["Close"]
        #closing_data = closing_data.fillna(method="ffill")[::-1].fillna(method="ffill") #original　なぜfillnaが2回あるのか[::-1]が不明
        closing_data = closing_data.fillna(method="ffill")
        #正規化
        for index in INDEX_LIST:
            if normalize:
                closing_data[index] = closing_data[index] / max(closing_data[index])
            if logreturn:
                closing_data[index] = np.log(closing_data[index] / closing_data[index].shift())
        return closing_data

    def calcorrelation(self,data,thvalue):
        '''
        baseリストと他リストとの相関係数算出用関数
        返り値として相関係数がthvalue以上の系列名を返却する
        '''
        effectivelist = []
        closing_data = data
        basevalue = INDEX_LIST[0]
        tmp = pd.DataFrame()
        tmp[basevalue] = closing_data[basevalue]
        for index in INDEX_LIST[1:]: 
            tmp[index] = closing_data[index]
            r = tmp.corr().iloc[:, 0]

            if(r.values[-1]>thvalue):
                effectivelist.append(r.index[-1])
        print(r)
        return effectivelist
    
    def mveffectivedata(self,effectivedata):
        '''
        相関係数がthvalue以上の系列のファイルを
        efdataディレクトリへコピーする
        '''
        files = os.listdir(self.basedir)
        files_file = [f for f in files if os.path.isfile(os.path.join(self.basedir, f))]
        for index in effectivedata:
            for filename in files_file:
                if( filename == index + ".csv"):
                    args = ['cp', self.basedir + filename, self.efdatadir]
                    try:
                        res = subprocess.check_call(args)
                        print("extract: ", index)
                    except:
                        print("command error.")

    def calcorrelation_plot(self,data):
        '''
        baseリストと他リストとの相関係数算出用関数
        '''
        closing_data = data
        corr_mat = closing_data.corr(method='pearson') 
        sns.heatmap(corr_mat,
            vmin=-1.0,
            vmax=1.0,
            center=0,
            annot=True, # True:格子の中に値を表示
            fmt='.1f',
            xticklabels=corr_mat.columns.values,
            yticklabels=corr_mat.columns.values
           )
        plt.show()

    def figureplot(self,data):
        '''
        経済指標データを可視化する
        '''
        closing_data.plot() #経済指標ごとの折れ線グラフ
        plt.legend(loc="upper right")
        plt.show()

    def tf_confusion_metrics(self,model, actual_classes, session, feed_dict):
        '''
        精度、再現性、正確度を計算する関数
        '''
        predictions = tf.argmax(model, 1)
        actuals = tf.argmax(actual_classes, 1)
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)

        tp_op = tf.reduce_sum(
            tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals), 
                tf.equal(predictions, ones_like_predictions)
            ), "float"
            )
        )

        tn_op = tf.reduce_sum(
            tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals), 
                tf.equal(predictions, zeros_like_predictions)
            ), "float"
            )
        )

        fp_op = tf.reduce_sum(
            tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals), 
                tf.equal(predictions, ones_like_predictions)
            ), "float"
            )
        )

        fn_op = tf.reduce_sum(
            tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals), 
                tf.equal(predictions, zeros_like_predictions)
            ),"float"
            )
        )

        tp, tn, fp, fn = \
            session.run(
            [tp_op, tn_op, fp_op, fn_op], 
            feed_dict
            )

        tpr = float(tp)/(float(tp) + float(fn))
        fpr = float(fp)/(float(tp) + float(fn))

        accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))

        recall = tpr
        precision = float(tp)/(float(tp) + float(fp))
  
        f1_score = (2 * (precision * recall)) / (precision + recall)
  
        print('Precision = ', precision)
        print('Recall = ', recall)
        print('F1 Score = ', f1_score)
        print('Accuracy = ', accuracy)

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

    def loss(self, model, actual_classes):
        '''
        誤差関数
        '''
        cost = -tf.reduce_sum(actual_classes*tf.log(model))
        return cost

    def training(self, cost,gstep, learning_rate=0.0001):
        '''
        学習
        '''
        training_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost,global_step=gstep)
        return training_step

if __name__ == '__main__':
    s = StockData()
    #s.getcsv() #株価データ取得
    closing_data = s.get_closing_data(normalize=False,logreturn=True)
    effectivelist = s.calcorrelation(closing_data,0.2) 
    s.mveffectivedata(effectivelist)
    #s.figureplot(closing_data)
   
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

    sess = tf.Session()
    num_predictors = len(training_predictors_tf.columns)
    num_classes = len(training_classes_tf.columns)

    feature_data = tf.placeholder("float", [None, num_predictors])
    actual_classes = tf.placeholder("float", [None, num_classes])

    #隠れ層情報をパラメータ化
    layers = [num_predictors,30,num_classes]
    global_step = tf.Variable(0, name='global_step', trainable=False)

    #model = s.inference(feature_data, num_predictors)
    model = s.inference(feature_data,layers)
    cost = s.loss(model, actual_classes)
    tf.summary.scalar('cost',cost)
    training_step = s.training(cost,global_step, learning_rate=0.0001)
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    #初期化
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep = 3)
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:

        ckpt_state = tf.train.get_checkpoint_state('models/')
        summary_writer = tf.summary.FileWriter(s.LOG_DIR,sess.graph)

        if ckpt_state:
            last_model = ckpt_state.model_checkpoint_path
            saver.restore(sess,last_model)
            print("model was loaded:", last_model)
        else:
            sess.run(init)
            print("initialized.")

        last_step = sess.run(global_step)
        for i in range(1, 10001):
            step = last_step + i 
            sess.run(
                training_step,
                feed_dict={
                    feature_data: training_predictors_tf.values,
                    actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
                }
            )
            if (step+1) % 100 == 0:
                summary, loss_, accuracy_  = sess.run([summaries, cost, accuracy],feed_dict={
                    feature_data: training_predictors_tf.values,
                    actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
                })
                summary_writer.add_summary(summary, (step+1))
                saver.save(sess,'models/mymodel',global_step=(step+1),write_meta_graph=False)

        feed_dict={
            feature_data: training_predictors_tf.values,
            actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
        }
        s.tf_confusion_metrics(model, actual_classes, sess, feed_dict)
    