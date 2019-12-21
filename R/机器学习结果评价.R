RF_result_test = read.csv('./RF_result_test.csv',header = T)
RF_result_valid = read.csv('./RF_result_valid.csv',header = T)
SVM_result_test = read.csv('./SVM_result_test.csv',header = T)
SVM_result_valid = read.csv('./SVM_result_valid.csv',header = T)

getfun = function(n = 3){
  mean1 = function(x,...){
    return(round(mean(x, na.rm = T),...))
  }
  sd1 = function(x,...){
    return(round(sd(x, na.rm = T),...))
  }
  dscp = function(x){
    return(paste0(mean1(x,digits=n),'    ',sd1(x)))
  }
  return(dscp)
}

dscp = getfun(4)
dscp(SVM_result_valid$auc)
