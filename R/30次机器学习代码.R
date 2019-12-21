library(caret) # »ìÏý¾ØÕó
library(pROC) # auc
library(randomForest)
library(e1071)

# getwd()
setwd('/home/ljw/project1/R/')

ds = read.csv('../data/data7/Xs.csv')
ds$y7 = factor(ds$y7)


unique.pid = unique(ds$pid)
n = length(unique.pid)

getdata = function(){
  index = sample(1:n, size=n, replace = F)
  cutoff1 = floor(n*0.7)
  cutoff2 = floor(n*0.9)
  train = ds[ds$pid %in% unique.pid[1:cutoff1], c(2:136)]
  test = ds[ds$pid %in% unique.pid[(cutoff1+1):cutoff2], c(2:136)]
  valid = ds[ds$pid %in% unique.pid[(cutoff2+1):n], c(2:136)]
  return(list(train, test, valid))
}


# RF --------------------------------------
rslt.te = list()
rslt.va = list()
for(i in 1:30){
  cat(i, 'is start ... \n')
  datas = getdata()
  train = datas[[1]]
  test = datas[[2]]
  valid = datas[[3]]
  
  cat('building model ... \n')
  fit<-randomForest(y7~., data=train)
  pred.te = predict(fit, newdata = test)
  pred.prob.te = predict(fit, newdata = test,type='prob')
  confm.te = confusionMatrix(pred.te, test$y7)
  acc.te = confm.te$overall['Accuracy']
  rslt.te[['acc']] = c(rslt.te[['acc']], acc.te)
  
  
  auc.te = auc(test$y7 ,pred.prob.te[,2])
  rslt.te[['auc']] = c(rslt.te[['auc']],auc.te)
  cat('test done ... \n')
  
  pred.va = predict(fit, newdata = valid)
  pred.prob.va = predict(fit, newdata = valid,type='prob')
  confm.va = confusionMatrix(pred.va, valid$y7)
  acc.va = confm.va$overall['Accuracy']
  rslt.va[['acc']] = c(rslt.va[['acc']], acc.va)
  
  auc.va = auc(valid$y7 ,pred.prob.va[,2])
  rslt.va[['auc']] = c(rslt.va[['auc']],auc.va)
  cat('valid done ... \n')
}


rslt.te = data.frame(rslt.te)
rslt.va = data.frame(rslt.va)

write.csv(rslt.te, './RF_result_test.csv', row.names = F)
write.csv(rslt.va, './RF_result_valid.csv', row.names = F)



# SVM - -----------------------

rslt.te = list()
rslt.va = list()
for(i in 1:30){
  cat(i, 'is start ... \n')
  datas = getdata()
  train = datas[[1]]
  test = datas[[2]]
  valid = datas[[3]]
  
  cat('building model ... \n')
  fit = svm(y7~., data=train,probability=T)
  pred.te.prob = predict(fit, newdata = test,probability = T)
  pred.te = attributes(pred.te.prob)['probabilities'][[1]]
  auc.te = auc(test$y7 ,pred.te[,2])
  confm.te = confusionMatrix(pred.te.prob, test$y7)
  acc.te = confm.te$overall['Accuracy']
  rslt.te[['acc']] = c(rslt.te[['acc']], acc.te)
  rslt.te[['auc']] = c(rslt.te[['auc']],auc.te)
  cat('test done ... \n')
  
  pred.va.prob = predict(fit, newdata = valid,probability = T)
  pred.va = attributes(pred.va.prob)['probabilities'][[1]]
  auc.va = auc(valid$y7 ,pred.va[,2])
  confm.va = confusionMatrix(pred.va.prob, valid$y7)
  acc.va = confm.va$overall['Accuracy']
  rslt.va[['acc']] = c(rslt.va[['acc']], acc.va)
  rslt.va[['auc']] = c(rslt.va[['auc']],auc.va)
  cat('valid done ... \n')
}


rslt.te = data.frame(rslt.te)
rslt.va = data.frame(rslt.va)

write.csv(rslt.te, './SVM_result_test.csv', row.names = F)
write.csv(rslt.va, './SVM_result_valid.csv', row.names = F)


