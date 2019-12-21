import pandas as pd
import numpy as np
import random
import os
import torch

class Dataloader:
	def __init__(self):
		pass

	def read(time_step=15, **kwargs):
		idvar = 'pid'
		varstartindex = 2
		timeindex = 1
		# print(os.getcwd())
		ds_Xs = pd.read_csv(open(kwargs['xs'], 'rb'), header=0)
		ds_mask = pd.read_csv(open(kwargs['ma'], 'rb'), header=0)
		ds_dt = pd.read_csv(open(kwargs['de'], 'rb'), header=0)

		# 需要 Xs, y, mask, timepoint, deltatime, x_mean, jump
		Xs = []  # (batch, timestep, var)
		y = []  # (batch, timestep, 1)
		mask = []  # (batch, timestep, var)
		timepoint = []  # (batch, timestep, 1)
		deltatime = []  # (batch, timestep, var)
		jump = []  # (batch, timestep, var)

		# 设定时间窗长度
		# time_step = 150

		count = 1  # 计数器
		# 遍历每一个 hamd_id
		for i in ds_Xs[idvar].unique():  # pd.dataframe ==> pd.series	unique(ds$pid)	iterator
			# i = ds_Xs['hadm_id'].unique()[1]
			# print(i)

			ds_Xs_temp = ds_Xs.loc[ds_Xs[idvar] == i, :]  # ds_Xs[ds_Xs$pid==i,] -> ds_Xs_temp
			ds_mask_temp = ds_mask.loc[ds_mask[idvar] == i, :]
			ds_dt_temp = ds_dt.loc[ds_dt[idvar] == i, :]
			# print(ds_Xs_temp.head())
			# print(ds_mask_temp.head())
			# print(ds_dt_temp.head())

			# ds_Xs_temp.keys().__len__()
			if ds_Xs_temp.shape[0] < time_step:  # dim(ds_temp)[1]   nrow(ds_temp)
				# 生成 Xs
				Xs_temp = np.asarray(ds_Xs_temp.iloc[:, varstartindex:-1])  # ds[, 2:length(ds)-1]
				Xs_temp = np.vstack((Xs_temp, np.tile(Xs_temp[-1, :], (
				time_step - Xs_temp.shape[0], 1))))  # rbind( ds, rep(ds[-1, ] , 15-10)  )
				Xs.append(Xs_temp)
				# 生成 y
				y_temp = np.asarray(ds_Xs_temp.iloc[:, -1]).reshape(-1, 1)
				y_temp = np.vstack((y_temp, np.tile(y_temp[-1, :], (time_step - y_temp.shape[0], 1))))
				y.append(y_temp)
				# 生成 mask
				mask_temp = np.asarray(ds_mask_temp.iloc[:, varstartindex:])
				mask_temp = np.vstack((mask_temp, np.tile(mask_temp[-1, :], (time_step - mask_temp.shape[0], 1))))
				mask.append(mask_temp)
				# 生成timepoint
				time_temp = np.asarray(ds_Xs_temp.iloc[:, timeindex]).reshape(-1, 1)
				time_temp = np.vstack((time_temp, np.tile(time_temp[-1, :], (time_step - time_temp.shape[0], 1))))
				timepoint.append(time_temp)
				# 生成deltatime
				dt_temp = np.asarray(ds_dt_temp.iloc[:, varstartindex:])
				dt_temp = np.vstack((dt_temp, np.tile(dt_temp[-1, :], (time_step - dt_temp.shape[0], 1))))
				deltatime.append(dt_temp)
				# 生成jump
				jump_temp = np.tile(1, (ds_Xs_temp.shape[0], 1))
				jump_temp = np.vstack((jump_temp, np.tile(0, (time_step - jump_temp.shape[0], 1))))
				jump.append(jump_temp)

			else:
				# 生成 Xs
				Xs_temp = np.asarray(ds_Xs_temp.iloc[(ds_Xs_temp.shape[0] - time_step):, varstartindex:-1])
				Xs.append(Xs_temp)
				# 生成 y
				y_temp = np.asarray(ds_Xs_temp.iloc[(ds_Xs_temp.shape[0] - time_step):, -1]).reshape(-1, 1)
				y.append(y_temp)
				# 生成 mask
				mask_temp = np.asarray(ds_mask_temp.iloc[(ds_mask_temp.shape[0] - time_step):, varstartindex:])
				mask.append(mask_temp)
				# 生成timepoint
				time_temp = np.asarray(ds_Xs_temp.iloc[(ds_Xs_temp.shape[0] - time_step):, timeindex]).reshape(-1, 1)
				timepoint.append(time_temp)
				# 生成deltatime
				dt_temp = np.asarray(ds_dt_temp.iloc[(ds_dt_temp.shape[0] - time_step):, varstartindex:])
				deltatime.append(dt_temp)
				# 生成jump
				jump_temp = np.tile(1, (time_step, 1))
				jump.append(jump_temp)
			if count % 100 == 0:
				print('{}: {} is done'.format(count, i))
			count += 1

		Xs = np.concatenate(Xs)
		y = np.concatenate(y)
		mask = np.concatenate(mask)
		timepoint = np.concatenate(timepoint)
		deltatime = np.concatenate(deltatime)
		jump = np.concatenate(jump)
		x_mean = np.mean(Xs, axis=0)  # (var,)

		varnum = Xs.shape[1]

		Xs = Xs.reshape(-1, time_step, varnum)
		y = y.reshape(-1, time_step, 1)
		mask = mask.reshape(-1, time_step, varnum)
		timepoint = timepoint.reshape(-1, time_step, 1)
		deltatime = deltatime.reshape(-1, time_step, varnum)
		jump = jump.reshape(-1, time_step, 1)
		x_mean = x_mean.reshape(1, -1)

		dataloader = Dataloader()
		dataloader.data = (Xs, y, mask, jump, timepoint, deltatime, x_mean)
		return dataloader

	def getdata(self, p=(0,7,0.9), seed=520):
		Xs, y, mask, jump, timepoint, deltatime, x_mean = self.data
		random.seed(seed)
		# 产生随机序列
		sample_index = random.sample(range(0, Xs.shape[0]), Xs.shape[0])
		cutoff1 = np.floor(Xs.shape[0] * p[0]).astype('int32')
		train_index = sample_index[0:cutoff1]
		# construct train
		Xs_train = Xs[train_index]
		y_train = y[train_index]
		mask_train = mask[train_index]
		jump_train = jump[train_index]
		timepoint_train = timepoint[train_index]
		deltatime_train = deltatime[train_index]
		if len(p) < 2:
			test_index = sample_index[(cutoff1 + 1):]
			# construct test
			Xs_test = Xs[test_index]
			y_test = y[test_index]
			mask_test = mask[test_index]
			jump_test = jump[test_index]
			timepoint_test = timepoint[test_index]
			deltatime_test = deltatime[test_index]
			return ((Xs_train, y_train, mask_train, jump_train, timepoint_train, deltatime_train, x_mean, train_index),
					(Xs_test, y_test, mask_test, jump_test, timepoint_test, deltatime_test, x_mean, test_index))
		else:
			cutoff2 = np.floor(Xs.shape[0] * p[1]).astype('int32')
			test_index = sample_index[(cutoff1 + 1):cutoff2]
			valid_index = sample_index[(cutoff2 + 1):]

			# construct test
			Xs_test = Xs[test_index]
			y_test = y[test_index]
			mask_test = mask[test_index]
			jump_test = jump[test_index]
			timepoint_test = timepoint[test_index]
			deltatime_test = deltatime[test_index]

			# construct validation
			Xs_valid = Xs[valid_index]
			y_valid = y[valid_index]
			mask_valid = mask[valid_index]
			jump_valid = jump[valid_index]
			timepoint_valid = timepoint[valid_index]
			deltatime_valid = deltatime[valid_index]
			return ((Xs_train, y_train, mask_train, jump_train, timepoint_train, deltatime_train, x_mean, train_index),
					(Xs_test, y_test, mask_test, jump_test, timepoint_test, deltatime_test, x_mean, test_index),
					(Xs_valid, y_valid, mask_valid, jump_valid, timepoint_valid, deltatime_valid, x_mean, valid_index))



if __name__ == "__main__":
	dataloader = Dataloader.read(time_step=10, xs='../data/Xs.csv',\
							ma='../data/mask.csv',de='../data/deltat.csv')
	data = dataloader.getdata((0.7,),520)
