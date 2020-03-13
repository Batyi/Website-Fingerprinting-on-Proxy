from scapy.all import *
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import support_functions as SUP
from numpy import array
import write_file as WF
import log_dataset_csv as csv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
import statistics
import os
from itertools import product
from sklearn.utils import check_matplotlib_support
from sklearn.base import is_classifier
from sklearn import preprocessing

def read_joined_df(file_name):
	path_to_file_name = SUP.merge_file_path_output(file_name)
	df = pd.read_csv(path_to_file_name)
	return df

def read_files(feature_names, urls_names):
	path = '/home/batyi/Desktop/Study/Study project/website_fingerprinting_proxy-stable/workspace/features_files/'
	sourcepath = os.listdir(path)
	# print(len(sourcepath))
	df_main = pd.DataFrame()
	# to measure maximum number if rows among files
	# max_row_length = 0
	# for name in feature_names:
	# 	for sfile in sourcepath:
	# 		if name+'_tls.csv' == sfile:
	# 			df = pd.read_csv(path + name+'_tls.csv')
	# 			if max_row_length < df.shape[0]:
	# 				max_row_length = df.shape[0]
	# print(max_row_length) 29021
	t = 0
	for name in feature_names:
		for sfile in sourcepath:
			if name+'_tls.csv' == sfile:
				df = pd.read_csv(path + name+'_tls.csv')
				# print(df.shape[0])
				# print(df['direction'])

				df = df.replace({'direction': 0}, -1)
				df_main = df_main.append(df['direction'],ignore_index=True,sort=False)
				t += 1
				break
		# if t > 1000:
		# 	break
	df_main = df_main.fillna(0)
	df_main = df_main.loc[:,0:4999]
	df_main['index'] = feature_names
	# df_main['index'] = feature_names[0:1001]
	

	#Rearrange cols in any way you want, moved the last element to the first position:
	cols = df_main.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	df_main = df_main[cols]

	df_main['site'] = urls_names
	# df_main['site'] = urls_names[0:1001]
	WF.write_csv(df_main, "cnn_dataframe.csv")
	print(df_main)


def merge_file_path(file_name):
  curr_dir = os.getcwd()
  get_file_path = os.path.join(curr_dir, "data_for_ML")
  print(file_name)
  print(get_file_path)
  abs_path = os.path.join(get_file_path, file_name)
  #print(abs_path)
  
  return abs_path

def merge_file_path_output(file_name):
  curr_dir = os.getcwd()
  #print(curr_dir)
  get_file_path = os.path.join(curr_dir, "output")
  #print(file_name)
  #print(get_file_path)
  abs_path = os.path.join(get_file_path, file_name)
  #print(abs_path)
  
  return abs_path

def prepare_data():
	file_name = "joined_df.csv"
	joined_df = read_joined_df(file_name)
	# print(joined_df['index'])
	feature_names = joined_df['index'].to_numpy()
	urls_names = joined_df['site'].to_numpy()
	read_files(feature_names, urls_names)
	# print(feature_names)
	

if __name__ == "__main__":
	prepare_data()
	pass

  