import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from utils.loaddata import Dataloader
from utils.functions import *
from models import *
import argparse
from modelinfo import MODELNAME
import os

parser = argparse.ArgumentParser()
parser.add_argument("--xs", type=str, default='./data/data7/Xs.csv',help="xs.csv")
parser.add_argument("--ma", type=str, default='./data/data7/mask.csv',help="mask.csv")
parser.add_argument("--de", type=str, default='./data/data7/deltat.csv',help="deltat.csv")
parser.add_argument("--epoch", type=int, default=200, help="The times of training")
parser.add_argument("--lr", type=float, default=0.001, help="learn rate")
parser.add_argument("--mt", type=int, default=1, help="model type")
parser.add_argument("--bat", type=int, default=200, help="batch size")
parser.add_argument("--tp", type=int, default=10, help="time step")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument("--bign", type=int, default=5, help="iter num")
parser.add_argument("--hid", type=int, default=200, help="hidden num")
parser.add_argument("--timew", type=int, default=7, help="timw window")
args = parser.parse_args()

dataloader = Dataloader.read(time_step=args.tp, xs=args.xs, ma=args.ma, de=args.de)
def getdata():
	seed = random.randint(0, 1e6)
	inputs = dataloader.getdata((0.7, 0.9), seed)
	((Xs_train, y_train, mask_train, jump_train, timepoint_train, deltatime_train, x_mean, train_index), \
	 (Xs_test, y_test, mask_test, jump_test, timepoint_test, deltatime_test, x_mean, test_index), \
	 (Xs_valid, y_valid, mask_valid, jump_valid, timepoint_valid, deltatime_valid, x_mean, valid_index)) = inputs

	Xs_train = torch.from_numpy(Xs_train).to(torch.float32)
	y_train = torch.from_numpy(y_train).to(torch.int64)
	mask_train = torch.from_numpy(mask_train).to(torch.float32)
	jump_train = torch.from_numpy(jump_train).to(torch.int64)
	deltatime_train = torch.from_numpy(deltatime_train).to(torch.float32)
	timepoint_train = torch.from_numpy(timepoint_train).to(torch.float32)
	x_mean = torch.from_numpy(x_mean).to(torch.float32)

	Xs_test = torch.from_numpy(Xs_test).to(torch.float32)
	# y_test = torch.from_numpy(y_test).to(torch.int64)
	mask_test = torch.from_numpy(mask_test).to(torch.float32)
	# jump_test = torch.from_numpy(jump_test).to(torch.int64)
	deltatime_test = torch.from_numpy(deltatime_test).to(torch.float32)
	timepoint_test = torch.from_numpy(timepoint_test).to(torch.float32)
	jump_test1 = jump_test

	Xs_valid = torch.from_numpy(Xs_valid).to(torch.float32)
	# y_valid = torch.from_numpy(y_valid).to(torch.int64)
	mask_valid = torch.from_numpy(mask_valid).to(torch.float32)
	# jump_valid = torch.from_numpy(jump_valid).to(torch.int64)
	deltatime_valid = torch.from_numpy(deltatime_valid).to(torch.float32)
	timepoint_valid = torch.from_numpy(timepoint_valid).to(torch.float32)
	jump_valid1 = jump_valid

	return ((Xs_train, y_train, mask_train, jump_train, timepoint_train, deltatime_train, x_mean, train_index), \
	 (Xs_test, y_test, mask_test, jump_test, timepoint_test, deltatime_test, x_mean, test_index), \
	 (Xs_valid, y_valid, mask_valid, jump_valid, timepoint_valid, deltatime_valid, x_mean, valid_index))

((Xs_train, y_train, mask_train, jump_train, timepoint_train, deltatime_train, x_mean, train_index), \
 (Xs_test, y_test, mask_test, jump_test, timepoint_test, deltatime_test, x_mean, test_index), \
 (Xs_valid, y_valid, mask_valid, jump_valid, timepoint_valid, deltatime_valid, x_mean, valid_index)) = getdata()


if __name__ == '__main__':

    modeltype = MODELNAME[args.mt]
    modeldef = MODEL[args.mt]

    indexs, N = getbatch(n=Xs_train.shape[0], batch=args.bat, shuffle=True)

    epoch = args.epoch

    docpath = './result_doc/' + modeltype
    if not os.path.exists(docpath):
        os.mkdir(docpath)
        docpath = docpath + '/timewindow'+str(args.timew)
        os.mkdir(docpath)
        f = open(docpath + '/model_eval.txt', 'a')
    else:
        docpath = docpath + '/timewindow' + str(args.timew)
        if not os.path.exists(docpath):
            os.mkdir(docpath)
            f = open(docpath+ '/model_eval.txt', 'a')
        else:
            f = open(docpath+ '/model_eval.txt', 'a')

    picpath = './result_pic/' + modeltype
    if not os.path.exists(picpath):
        os.mkdir(picpath)
        picpath = picpath + '/timewindow'+str(args.timew)
        os.mkdir(picpath)
    else:
        picpath = picpath + '/timewindow' + str(args.timew)
        if not os.path.exists(picpath):
            os.mkdir(picpath)

    for k in range(args.bign):
        model = modeldef(x_mean=x_mean, \
                        input_size=Xs_train.shape[-1], \
                        timestep=args.tp, \
                        hidden_size=args.hid)

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        losses, accs_te, preses_te, recalls_te, aucs_te, f1s_te = [], [], [], [], [], []
        accs_va, preses_va, recalls_va, aucs_va, f1s_va = [], [], [], [], []

        # max_acc_test, max_acc_valid = 0.0, 0.0
        # max_auc_test, max_auc_valid = 0.0, 0.0
        acc_tests, acc_valids = [], []
        auc_tests, auc_valids = [], []

        ((Xs_train, y_train, mask_train, jump_train, timepoint_train, deltatime_train, x_mean, train_index), \
         (Xs_test, y_test, mask_test, jump_test, timepoint_test, deltatime_test, x_mean, test_index), \
         (Xs_valid, y_valid, mask_valid, jump_valid, timepoint_valid, deltatime_valid, x_mean, valid_index)) = getdata()
        for i in range(epoch):
            indexs, N = getbatch(n=Xs_train.shape[0], batch=args.bat, shuffle=True)
            xhat_trs = []
            for j in range(N):
                output_tr, xhat_tr = model(Xs_train[indexs[j]], deltatime_train[indexs[j]], mask_train[indexs[j]],
                                           timepoint_train[indexs[j]])
                if i == epoch - 1:
                    xhat_trs.extend(torch.stack(xhat_tr, dim=1))
                output_tr = torch.stack(output_tr, dim=1)
                output_tr1 = output_tr.view(-1, 2)
                output_tr2 = torch.argmax(output_tr, dim=1)

                jump1_train = jump_train[indexs[j]].view(-1, ).data.numpy()

                loss = loss_func(output_tr1[jump1_train == 1], y_train[indexs[j]].view(-1, )[jump1_train == 1]) / \
                       Xs_train[indexs[j]].shape[0]

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            print("train loss: {}".format(loss))
            with torch.no_grad():
                output_te, xhat_te = model(Xs_test, deltatime_test, mask_test, timepoint_test)
                output_te = torch.stack(output_te, dim=1)
                output_te_out = output_te
                output_te2 = output_te.view(-1, 2).data.numpy()
                output_te1 = output_te.view(-1, 2).data.numpy()
                output_te1 = np.argmax(output_te1, axis=1)
                jump_test = jump_test.reshape(-1, )

                acc_test = accuracy_score(y_test.reshape(-1, )[jump_test == 1], output_te1[jump_test == 1])
                pres_test = precision_score(y_test.reshape(-1, )[jump_test == 1], output_te1[jump_test == 1])
                recal_test = recall_score(y_test.reshape(-1, )[jump_test == 1], output_te1[jump_test == 1])
                auc_test = roc_auc_score(y_test.reshape(-1, )[jump_test == 1], output_te2[jump_test == 1, 1])
                f1_test = f1_score(y_test.reshape(-1, )[jump_test == 1], output_te1[jump_test == 1])


                print("test: epoch: {}, acc: {}, pres: {}, recall: {}, auc: {}, f1: {}".format(i + 1, acc_test, pres_test,\
                                                                                               recal_test, auc_test, \
                                                                                               f1_test))
                accs_te.append(acc_test)
                preses_te.append(pres_test)
                recalls_te.append(recal_test)
                aucs_te.append(auc_test)
                f1s_te.append(f1_test)
                losses.append(loss)

                fig = plt.figure()
                plt.plot(accs_te, label='Accuracy')
                plt.plot(preses_te, label='Precision')
                plt.plot(recalls_te, label='Recall')
                plt.plot(aucs_te, label='AUROC')
                plt.plot(f1s_te, label='F1-measure')
                plt.legend()
                plt.savefig(picpath+'/Test'+str(k)+'Rslt.png')
                plt.close()

                fig = plt.figure()
                plt.plot(losses, label='Train Loss')
                plt.legend()
                plt.savefig(picpath +'/Train'+str(k)+'Loss')
                plt.close()

                output_va, xhat_va = model(Xs_valid, deltatime_valid, mask_valid, timepoint_valid)
                output_va = torch.stack(output_va, dim=1)
                output_va_out = output_va
                output_va2 = output_va.view(-1, 2).data.numpy()
                output_va1 = output_va.view(-1, 2).data.numpy()
                output_va1 = np.argmax(output_va1, axis=1)
                jump_valid = jump_valid.reshape(-1, )

                acc_valid = accuracy_score(y_valid.reshape(-1, )[jump_valid == 1], output_va1[jump_valid == 1])
                pres_valid = precision_score(y_valid.reshape(-1, )[jump_valid == 1], output_va1[jump_valid == 1])
                recal_valid = recall_score(y_valid.reshape(-1, )[jump_valid == 1], output_va1[jump_valid == 1])
                auc_valid = roc_auc_score(y_valid.reshape(-1, )[jump_valid == 1], output_va2[jump_valid == 1, 1])
                f1_valid = f1_score(y_valid.reshape(-1, )[jump_valid == 1], output_va1[jump_valid == 1])

                accs_va.append(acc_valid)
                preses_va.append(pres_valid)
                recalls_va.append(recal_valid)
                aucs_va.append(auc_valid)
                f1s_va.append(f1_valid)

                plt.figure()
                plt.plot(accs_va, label='Accuracy')
                plt.plot(preses_va, label='Precisin')
                plt.plot(recalls_va, label='Recall')
                plt.plot(aucs_va, label='AUROC')
                plt.plot(f1s_va, label='F1-measure')
                plt.legend()
                plt.savefig(picpath +'/Valid' + str(k) + 'Rslt.png')
                plt.close()

                print("valid: epoch: {}, acc: {}, pres: {}, recall: {}, auc: {}, f1: {}".format(i + 1, acc_valid, pres_valid,\
                                                                                                 recal_valid, auc_valid,\
                                                                                                 f1_valid))

            acc_tests.append(str(accs_te[-1]))
            auc_tests.append(str(aucs_te[-1]))
            acc_valids.append(str(accs_va[-1]))
            auc_valids.append(str(aucs_va[-1]))

        f.write(str(k)+': acc_test: '+','.join(acc_tests)+'\n')
        f.write(str(k)+': auc_test: '+','.join(auc_tests)+'\n')
        f.write(str(k)+': acc_valid: '+','.join(acc_valids)+'\n')
        f.write(str(k)+': auc_valid: '+','.join(auc_valids)+'\n')

    f.close()


