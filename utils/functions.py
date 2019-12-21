import numpy as np
import random

def cc(ds):
	return np.concatenate(ds, axis=0)

def getbatch(n, batch, shuffle=True):
	"""
	接受一个  N, batch, shuffle, 返回一个2层列表，和batch总数
	"""
	indexs = list(range(n))
	if shuffle:
		rdm_indexs = random.sample(indexs, len(indexs))
	else:
		rdm_indexs = indexs

	count = n // batch # 循环要进行的次数
	batches = [rdm_indexs[(i*batch):((i+1)*batch)] for i in range(count)]
	batches.append(rdm_indexs[((count)*batch):n])

	return batches, count+1