setwd('/home/ljw/project1/data/')


# 为每个时间窗创建文件夹
for(i in 3:10){
  timewindow = i
  if(!file.exists(paste0('./data', timewindow))){
    dir.create(paste0('./data', timewindow))
  }
}

# 修改每个时间窗里面数据集的名字统一为 data.csv
for(i in 3:10){
  timewindow = i
  fpath = paste0('data',timewindow,'/data_', timewindow,'.csv')
  fpath1 = paste0('data',timewindow,'/data.csv')
  file.rename(fpath, fpath1)
}



library(data.table)
for(i in 3:10){
  timewindow = i
  ds<-fread(file=paste0('./data',timewindow,'/data.csv'),
            header = T, fill=T, stringsAsFactors = F,
            sep=',', nThread = 50)
  Xs_gen<-ds[,1:(length(ds)-1)]
  # View(Xs_gen)
  fwrite(x=Xs_gen, file=paste0('./data',timewindow,'/Xs_gen.csv'), row.names = F, sep = ',', nThread = 50)
  Xs<-ds
  for (i in 1:length(Xs)) {
    Xs[[i]][is.na(Xs[[i]])]<-0
  }
  fwrite(x=Xs, file=paste0('./data',timewindow,'/Xs.csv'), row.names = F, sep = ',', nThread = 50)
}

