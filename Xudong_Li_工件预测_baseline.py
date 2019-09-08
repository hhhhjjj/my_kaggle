import numpy as np
import pandas as pd
import tensorflow as tf
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
import catboost as cbt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import copy
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
''''''
config_ = tf.ConfigProto()
config_.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
config_.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config_)
with tf.Session(config=config_) as sess:
	pass
# -----------------------------------------------------------------------
# read data
# -----------------------------------------------------------------------
train_data = pd.read_csv('./data/first_round_training_data.csv')
test_data = pd.read_csv('./data/first_round_testing_data.csv', encoding='utf-8')
dit = {'Excellent':0,'Good':1,'Pass':2,'Fail':3}
train_data['Quality_label'] = train_data['Quality_label'].map(dit)
labels = pd.get_dummies(train_data['Quality_label']).values
submit = pd.read_csv('./data/submit_example.csv')
# ------------------------------------------------------------------------
# 特征工程
# ------------------------------------------------------------------------
# 取parameter5-10，取log
features = ['Parameter5', 'Parameter6', 'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']
train_data[features] = np.log(train_data[features] + 1e-5)
test_data[features] = np.log(test_data[features] + 1e-5)
# 特征融合
train_data['5_6'] = train_data['Parameter5'] + train_data['Parameter6']
test_data['5_6'] = test_data['Parameter5'] + test_data['Parameter6']
train_data['5_7'] = train_data['Parameter5'] + train_data['Parameter7']
test_data['5_7'] = test_data['Parameter5'] + test_data['Parameter7']
train_data['7_8'] = train_data['Parameter7'] + train_data['Parameter8']
test_data['7_8'] = test_data['Parameter7'] + test_data['Parameter8']
train_data['7_9'] = train_data['Parameter7'] + train_data['Parameter9']
test_data['7_9'] = test_data['Parameter7'] + test_data['Parameter9']
train_data['7_10'] = train_data['Parameter7'] + train_data['Parameter10']
test_data['7_10'] = test_data['Parameter7'] + test_data['Parameter10']
train_data['8_9'] = train_data['Parameter8'] + train_data['Parameter9']
test_data['8_9'] = test_data['Parameter8'] + test_data['Parameter9']
train_data['8_10'] = train_data['Parameter8'] + train_data['Parameter10']
test_data['8_10'] = test_data['Parameter8'] + test_data['Parameter10']
train_data['9_10'] = train_data['Parameter9'] + train_data['Parameter10']
test_data['9_10'] = test_data['Parameter9'] + test_data['Parameter10']

train_data['5_10'] = train_data['Parameter5'] + train_data['Parameter10']
test_data['5_10'] = test_data['Parameter5'] + test_data['Parameter10']
train_data['6_10'] = train_data['Parameter6'] + train_data['Parameter10']
test_data['6_10'] = test_data['Parameter6'] + test_data['Parameter10']
train_data['7_8_9'] = train_data['Parameter7'] + train_data['Parameter8'] + train_data['Parameter9']
test_data['7_8_9'] = test_data['Parameter7'] + test_data['Parameter8'] + test_data['Parameter9']
train_data['7_9_10'] = train_data['Parameter7'] + train_data['Parameter9'] + train_data['Parameter10']
test_data['7_9_10'] = test_data['Parameter7'] + test_data['Parameter9'] + test_data['Parameter10']
train_data['7_8_10'] = train_data['Parameter7'] + train_data['Parameter8'] + train_data['Parameter10']
test_data['7_8_10'] = test_data['Parameter7'] + test_data['Parameter8'] + test_data['Parameter10']
train_data['8_9_10'] = train_data['Parameter8'] + train_data['Parameter9'] + train_data['Parameter10']
test_data['8_9_10'] = test_data['Parameter8'] + test_data['Parameter9'] + test_data['Parameter10']
train_data['7_8_9_10'] = train_data['Parameter7'] + train_data['Parameter8'] + train_data['Parameter9'] + train_data['Parameter10']
test_data['7_8_9_10'] = test_data['Parameter7'] + test_data['Parameter8'] + test_data['Parameter9'] + test_data['Parameter10']

train_data['5-6'] = (train_data['Parameter5'] - train_data['Parameter6'])
test_data['5-6'] = (test_data['Parameter5'] - test_data['Parameter6'])
train_data['7-8'] = (train_data['Parameter7'] - train_data['Parameter8'])
test_data['7-8'] = (test_data['Parameter7'] - test_data['Parameter8'])
train_data['7-9'] = (train_data['Parameter7'] - train_data['Parameter9'])
test_data['7-9'] = (test_data['Parameter7'] - test_data['Parameter9'])
train_data['7-10'] = (train_data['Parameter7'] - train_data['Parameter10'])
test_data['7-10'] = (test_data['Parameter7'] - test_data['Parameter10'])
train_data['8-9'] = (train_data['Parameter8'] - train_data['Parameter9'])
test_data['8-9'] = (test_data['Parameter8'] - test_data['Parameter9'])
train_data['8-10'] = (train_data['Parameter8'] - train_data['Parameter10'])
test_data['8-10'] = (test_data['Parameter8'] - test_data['Parameter10'])
train_data['9-10'] = (train_data['Parameter9'] - train_data['Parameter10'])
test_data['9-10'] = (test_data['Parameter9'] - test_data['Parameter10'])

train_data['5_6_mul'] = train_data['Parameter5'].mul(train_data['Parameter6'])
test_data['5_6_mul'] = test_data['Parameter5'].mul(test_data['Parameter6'])
train_data['7_8_mul'] = train_data['Parameter7'].mul(train_data['Parameter8'])
test_data['7_8_mul'] = test_data['Parameter7'].mul(test_data['Parameter8'])
train_data['7_9_mul'] = train_data['Parameter7'].mul(train_data['Parameter9'])
test_data['7_9_mul'] = test_data['Parameter7'].mul(test_data['Parameter9'])
train_data['7_10_mul'] = train_data['Parameter7'].mul(train_data['Parameter10'])
test_data['7_10_mul'] = test_data['Parameter7'].mul(test_data['Parameter10'])
train_data['8_9_mul'] = train_data['Parameter8'].mul(train_data['Parameter9'])
test_data['8_9_mul'] = test_data['Parameter8'].mul(test_data['Parameter9'])
train_data['8_10_mul'] = train_data['Parameter8'].mul(train_data['Parameter10'])
test_data['8_10_mul'] = test_data['Parameter8'].mul(test_data['Parameter10'])
train_data['9_10_mul'] = train_data['Parameter9'].mul(train_data['Parameter10'])
test_data['9_10_mul'] = test_data['Parameter9'].mul(test_data['Parameter10'])

train_data['78_mul_79'] = (train_data['7_8'].mul(train_data['7_9']))
test_data['78_mul_79'] = (test_data['7_8'].mul(test_data['7_9']))
train_data['78_mul_710'] = (train_data['7_8'].mul(train_data['7_10']))
test_data['78_mul_710'] = (test_data['7_8'].mul(test_data['7_10']))
train_data['78_mul_89'] = (train_data['7_8'].mul(train_data['8_9']))
test_data['78_mul_89'] = (test_data['7_8'].mul(test_data['8_9']))
train_data['78_mul_810'] = (train_data['7_8'].mul(train_data['8_10']))
test_data['78_mul_810'] = (test_data['7_8'].mul(test_data['8_10']))
train_data['78_mul_910'] = (train_data['7_8'].mul(train_data['9_10']))
test_data['78_mul_910'] = (test_data['7_8'].mul(test_data['9_10']))
train_data['79_mul_710'] = (train_data['7_9'].mul(train_data['7_10']))
test_data['79_mul_710'] = (test_data['7_9'].mul(test_data['7_10']))
train_data['79_mul_89'] = (train_data['7_9'].mul(train_data['8_9']))
test_data['79_mul_89'] = (test_data['7_9'].mul(test_data['8_9']))
train_data['79_mul_810'] = (train_data['7_9'].mul(train_data['8_10']))
test_data['79_mul_810'] = (test_data['7_9'].mul(test_data['8_10']))
train_data['79_mul_910'] = (train_data['7_9'].mul(train_data['9_10']))
test_data['79_mul_910'] = (test_data['7_9'].mul(test_data['9_10']))
train_data['710_mul_89'] = (train_data['7_10'].mul(train_data['8_9']))
test_data['710_mul_89'] = (test_data['7_10'].mul(test_data['8_9']))
train_data['710_mul_810'] = (train_data['7_10'].mul(train_data['8_10']))
test_data['710_mul_810'] = (test_data['7_10'].mul(test_data['8_10']))
train_data['710_mul_910'] = (train_data['7_10'].mul(train_data['9_10']))
test_data['710_mul_910'] = (test_data['7_10'].mul(test_data['9_10']))
train_data['89_mul_810'] = (train_data['8_9'].mul(train_data['8_10']))
test_data['89_mul_810'] = (test_data['8_9'].mul(test_data['8_10']))
train_data['89_mul_910'] = (train_data['8_9'].mul(train_data['9_10']))
test_data['89_mul_910'] = (test_data['8_9'].mul(test_data['9_10']))
train_data['810_mul_910'] = (train_data['8_10'].mul(train_data['9_10']))
test_data['810_mul_910'] = (test_data['8_10'].mul(test_data['9_10']))

train_data['5_8'] = train_data['Parameter5'] + train_data['Parameter8']
test_data['5_8'] = test_data['Parameter5'] + test_data['Parameter8']
train_data['7+8*8'] = train_data['Parameter7'] + train_data['Parameter8'] * train_data['Parameter8']
test_data['7+8*8'] = test_data['Parameter7'] + test_data['Parameter8'] * test_data['Parameter8']
train_data['7*7+8'] = train_data['Parameter7'] * train_data['Parameter7'] + train_data['Parameter8']
test_data['7*7+8'] = test_data['Parameter7'] * test_data['Parameter7'] + test_data['Parameter8']

train_data['9/7'] = train_data['Parameter9'] / train_data['Parameter7']
test_data['9/7'] = test_data['Parameter9'] / test_data['Parameter7']
train_data['9/8'] = train_data['Parameter9'] / train_data['Parameter8']
test_data['9/8'] = test_data['Parameter9'] / test_data['Parameter8']
train_data['1/7'] = 1. / train_data['Parameter7']
test_data['1/7'] = 1. / test_data['Parameter7']
train_data['9%8'] = train_data['Parameter9'] % train_data['Parameter8']
test_data['9%8'] = test_data['Parameter9'] % test_data['Parameter8']
train_data['7%9'] = train_data['Parameter7'] % train_data['Parameter9']
test_data['7%9'] = test_data['Parameter7'] % test_data['Parameter9']
train_data['1/9'] = 1. / train_data['Parameter9']
test_data['1/9'] = 1. / test_data['Parameter9']
train_data['7/8'] = train_data['Parameter7'] / train_data['Parameter8']
test_data['7/8'] = test_data['Parameter7'] / test_data['Parameter8']
train_data['8%9'] = train_data['Parameter8'] % train_data['Parameter9']
test_data['8%9'] = test_data['Parameter8'] % test_data['Parameter9']
train_data['7/9'] = train_data['Parameter7'] / train_data['Parameter9']
test_data['7/9'] = test_data['Parameter7'] / test_data['Parameter9']
train_data['8/7'] = train_data['Parameter8'] / train_data['Parameter7']
test_data['8/7'] = test_data['Parameter8'] / test_data['Parameter7']
train_data['1/8'] = 1. / train_data['Parameter8']
test_data['1/8'] = 1. / test_data['Parameter8']
train_data['8%7'] = train_data['Parameter8'] % train_data['Parameter7']
test_data['8%7'] = test_data['Parameter8'] % test_data['Parameter7']
train_data['8+9/8%9'] = (train_data['Parameter8'] + train_data['Parameter9']) / train_data['Parameter8'] % train_data['Parameter9']
test_data['8+9/8%9'] = (test_data['Parameter8'] + test_data['Parameter9']) / test_data['Parameter8'] % test_data['Parameter9']
train_data['7%8'] = train_data['Parameter7'] % train_data['Parameter8']
test_data['7%8'] = test_data['Parameter7'] % test_data['Parameter8']
train_data['8/9'] = train_data['Parameter8'] / train_data['Parameter9']
test_data['8/9'] = test_data['Parameter8'] / test_data['Parameter9']

train_data['7+8%7+9'] = train_data['Parameter7'] + train_data['Parameter8'] % train_data['Parameter7'] + train_data['Parameter9']
test_data['7+8%7+9'] = test_data['Parameter7'] + test_data['Parameter8'] % test_data['Parameter7'] + test_data['Parameter9']
train_data['1/8/9'] = 1./train_data['Parameter8']/train_data['Parameter9']
test_data['1/8/9'] = 1./test_data['Parameter8']/test_data['Parameter9']
train_data['9%7/9'] = train_data['Parameter9']%train_data['Parameter7']/train_data['Parameter9']
test_data['9%7/9'] = test_data['Parameter9']%test_data['Parameter7']/test_data['Parameter9']
train_data['8+9/8*9'] = train_data['Parameter8']+train_data['Parameter9']/train_data['Parameter8']*train_data['Parameter9']
test_data['8+9/8*9'] = test_data['Parameter8']+test_data['Parameter9']/test_data['Parameter8']*test_data['Parameter9']
train_data['9%7/7*8'] = train_data['Parameter9']%train_data['Parameter7']/train_data['Parameter7']*train_data['Parameter8']
test_data['9%7/7*8'] = test_data['Parameter9']%test_data['Parameter7']/test_data['Parameter7']*test_data['Parameter8']
train_data['7%8/7'] = train_data['Parameter7']%train_data['Parameter8']/train_data['Parameter7']
test_data['7%8/7'] = test_data['Parameter7']%test_data['Parameter8']/test_data['Parameter7']
train_data['9%8*9'] = train_data['Parameter9']%train_data['Parameter8']*train_data['Parameter9']
test_data['9%8*9'] = test_data['Parameter9']%test_data['Parameter8']*test_data['Parameter9']
train_data['8*9%7'] = train_data['Parameter8']*train_data['Parameter9']%train_data['Parameter7']
test_data['8*9%7'] = test_data['Parameter8']*test_data['Parameter9']%test_data['Parameter7']

train_data['8%9*8'] = train_data['Parameter8']%train_data['Parameter9']*train_data['Parameter8']
test_data['8%9*8'] = test_data['Parameter8']%test_data['Parameter9']*test_data['Parameter8']
train_data['7*8*9'] = train_data['Parameter8']*train_data['Parameter7']*train_data['Parameter9']
test_data['7*8*9'] = test_data['Parameter8']*test_data['Parameter7']*test_data['Parameter9']
train_data['8+9/7*8']=train_data['Parameter8']+train_data['Parameter9']/train_data['Parameter7']*train_data['Parameter9']
test_data['8+9/7*8']=test_data['Parameter8']+test_data['Parameter9']/test_data['Parameter7']*test_data['Parameter9']
train_data['8%7/(9%8)'] = train_data['Parameter8']%train_data['Parameter7']/(train_data['Parameter9']%train_data['Parameter8'])
test_data['8%7/(9%8)'] = test_data['Parameter8']%test_data['Parameter7']/(test_data['Parameter9']%test_data['Parameter8'])
train_data['7%8/(8%9)'] = train_data['Parameter7']%train_data['Parameter8']/(train_data['Parameter8']%train_data['Parameter9'])
test_data['7%8/(8%9)'] = test_data['Parameter7']%test_data['Parameter8']/(test_data['Parameter8']%test_data['Parameter9'])
train_data['8+9/(9%8)']=train_data['Parameter8']+train_data['Parameter9']/(train_data['Parameter9']%train_data['Parameter8'])
test_data['8+9/(9%8)']=test_data['Parameter8']+test_data['Parameter9']/(test_data['Parameter9']%test_data['Parameter8'])
train_data['8*(9%7)+8'] = train_data['Parameter8']*(train_data['Parameter9']%train_data['Parameter7']+train_data['Parameter8'])
test_data['8*(9%7)+8'] = test_data['Parameter8']*(test_data['Parameter9']%test_data['Parameter7']+test_data['Parameter8'])
train_data['7+8*9'] = train_data['Parameter7']+train_data['Parameter8']*train_data['Parameter9']
test_data['7+8*9'] = test_data['Parameter7']+test_data['Parameter8']*test_data['Parameter9']
train_data['7%8/(9%8)'] = train_data['Parameter7']%train_data['Parameter8']/(train_data['Parameter9']%train_data['Parameter8'])
test_data['7%8/(9%8)'] = test_data['Parameter7']%test_data['Parameter8']/(test_data['Parameter9']%test_data['Parameter8'])
train_data['9%8/(7%8)'] = train_data['Parameter9']%train_data['Parameter8']/(train_data['Parameter7']%train_data['Parameter8'])
test_data['9%8/(7%8)'] = test_data['Parameter9']%test_data['Parameter8']/(test_data['Parameter7']%test_data['Parameter8'])
train_data['5%8'] = train_data['Parameter5']%train_data['Parameter8']
test_data['5%8'] = test_data['Parameter5']%test_data['Parameter8']
train_data['6%8'] = train_data['Parameter6']%train_data['Parameter8']
test_data['6%8'] = test_data['Parameter6']%test_data['Parameter8']
train_data['10%8'] = train_data['Parameter10']%train_data['Parameter8']
test_data['10%8'] = test_data['Parameter10']%test_data['Parameter8']

train_data['7*8'] = train_data['Parameter7'] * train_data['Parameter8']
test_data['7*8'] = test_data['Parameter7'] * test_data['Parameter8']
train_data['5*5'] = train_data['Parameter5'] * train_data['Parameter5']
test_data['5*5'] = test_data['Parameter5'] * test_data['Parameter5']
train_data['6*6'] = train_data['Parameter6'] * train_data['Parameter6']
test_data['6*6'] = test_data['Parameter6'] * test_data['Parameter6']
train_data['7*7'] = train_data['Parameter7'] * train_data['Parameter7']
test_data['7*7'] = test_data['Parameter7'] * test_data['Parameter7']
train_data['8*8'] = train_data['Parameter8'] * train_data['Parameter8']
test_data['8*8'] = test_data['Parameter8'] * test_data['Parameter8']
train_data['9*9'] = train_data['Parameter9'] * train_data['Parameter9']
test_data['9*9'] = test_data['Parameter9'] * test_data['Parameter9']
train_data['10*10'] = train_data['Parameter10'] * train_data['Parameter10']
test_data['10*10'] = test_data['Parameter10'] * test_data['Parameter10']

scale_features = ['5_6', '7_9', '8_9', '7_8', '8_10', '9_10', '7_10', '9%8', '7/8', '7%8', '8/9',
				  ]


# 组合所有特征
features = features + scale_features

print('all train features:', features)

for fc in features:
	n = train_data['{}'.format(fc)].nunique()
	print(fc + ':', n)
# --------------------------------------------------------
# train
# --------------------------------------------------------
x_train = train_data[features].values
x_test = test_data[features].values

def lightgbm_model():
	model = LGBMClassifier(max_depth=7, learning_rate=0.01, n_estimators=1000, num_leaves=16,
						   objective='multiclass', silent=True,
						   reg_lambda=4., #reg_alpha=1.,
						   #bagging_fraction=0.9, feature_fraction=0.9,
						   )
	return model

def xgboost_model():
	model = XGBClassifier(max_depth=5, n_estimators=1000, learning_rate=0.01, silent=True,
						  #objective='multi:softmax',
						  )
	return model

def catboost_model():
	cbt_model = cbt.CatBoostClassifier(#iterations=100000, learning_rate=0.01, verbose=0,
		iterations=3000, learning_rate=0.01, verbose=0,
		max_depth=7, #reg_lambda=5.,
		task_type='GPU',   # cat_features=cat_list,
		loss_function='MultiClass')
	return cbt_model

def catboost_importance():
	model = catboost_model()
	model.fit(x_train, np.argmax(labels, 1))
	importance = model.get_feature_importance(prettified=True)
	print(importance.columns)
	for i in range(len(features)):
		print(features[int(importance['Feature Index'][i])] + ' :', importance['Importances'][i])
	
def grid_search(mode):
	def my_custom_loss_func(y_true, y_pred):
		diff = np.mean(-y_true * np.log(y_pred + 1e-5))
		return diff
	score = make_scorer(my_custom_loss_func, greater_is_better=False)
	
	# grid search for lightgbm
	if mode=='lgb':
		parameters = {'max_depth': [5, 7, 9, 11], 'num_leaves ': [16, 32, 64], 'reg_lambda': [10, 5, 3, 1, 0.1]}
		model = lightgbm_model()
	# grid search for catboost
	if mode=='cat':
		parameters = {'max_depth': [5, 7, 9, 11], 'l2_leaf_reg': [1., 2., 3., 4., 5.]}
		model = catboost_model()
	# grid search for xgboost
	if mode=='xgb':
		parameters = {'max_depth': [5, 7, 9, 11]}
		model = catboost_model()
	# search
	clf = GridSearchCV(model, parameters, cv=5, scoring=score, verbose=2)
	clf.fit(x_train, np.argmax(labels, 1))
	print('best parameters:', clf.best_params_)
	print('best score:', clf.best_score_)

catboost_importance()
# k-fold train
def kfold_train(mode):
	acc_list, loss_list = [], []
	prediction = np.zeros((x_test.shape[0], 4))
	for i in range(10):
		print(str(i+1) + ' th kflod' + '*'*50)
		kf = KFold(n_splits=5, shuffle=True, random_state=i)
		kfold_list = []
		for k, (train_index, test_index) in enumerate(kf.split(x_train)):
			print(str(k+1) + 'fold--------------')
			train_x, train_y = x_train[train_index], labels[train_index]
			test_x, test_y = x_train[test_index], labels[test_index]
			# train
			if mode == 'cat':
				model = catboost_model()
				model.fit(train_x, np.argmax(train_y, 1), eval_set=(test_x, np.argmax(test_y, 1)),
					  #early_stopping_rounds=1000, verbose=False
						)
				#print(pd.DataFrame({'column': features, 'importance': model.feature_importances_}).sort_values(by='importance'))
			if mode == 'lgb':
				model = lightgbm_model()
				model.fit(train_x, np.argmax(train_y, 1), eval_set=(test_x, np.argmax(test_y, 1)),
					  # early_stopping_rounds=50, verbose=True
						  verbose=False
						  )
			if mode == 'xgb':
				model = xgboost_model()
				model.fit(train_x, np.argmax(train_y, 1), verbose=True)
			# test
			pred = model.predict_proba(test_x)
			acc = accuracy_score(np.argmax(test_y, 1), np.argmax(pred, 1))
			loss = log_loss(test_y, pred)
			acc_list.append(acc)
			loss_list.append(loss)
			kfold_list.append(loss)
			print('test acc: %f, test loss: %f' % (acc, loss))
			# predict
			prediction += model.predict_proba(x_test)
		print('this fold mean loss:', np.mean(kfold_list))
	print('*'*50)
	print('mean acc: %f, mean loss: %f' % (np.mean(acc_list), np.mean(loss_list)))
	prediction = prediction / 50.
	return prediction

def submit_result(prediction):
	sub = test_data[['Group']]
	prob_cols = [i for i in submit.columns if i not in ['Group']]
	for i, f in enumerate(prob_cols):
		sub[f] = prediction[:, i]
	for i in prob_cols:
		sub[i] = sub.groupby('Group')[i].transform('mean')
	sub = sub.drop_duplicates()
	sub.to_csv("submission2.csv", index=False)

time1 = time.clock()
prediction = kfold_train('cat')
time2 = time.clock()
print('running time: ', str((time2 - time1)/60))
submit_result(prediction)
'''
grid_search('xgb')
'''