# T-D-PLSTM

This is a pytorch version of the code for a quick implementation of the method mentioned in the article "Predicting Timing of Surgical Intervention Using Recurrent Neural Network for Necrotizing Pancreatitis".This code contains data preprocessing and data modeling analysis.

These methods include **lstm-d, gru-d, plstm-d, Time weighted lstm-d, Time weighted gru-d, Time weighted plstm-d**

### Data prepoccessing
1. Firstly, your data should be like below:

2. Then unzip the data file

```bash
cd /data
bash unzip.sh
```

3. Run the R code to produce the intermediate data that needs to be used later
(Here, you need a R environment where the vision is greater than 3.3.0)
run **data_prepare.R** in an environment of R, maybe you will do this in Rstudio or the R's own IDE.

4. Create the folders for saving result pictures and docs.

```bash
cd ./project
bash create_result_folder.sh
```

### Modeling
1. Edit the --xs, --ma, --de, they represent the path of Xs.csv, mask.csv and deltat.csv, respectively. For the meaning of the datasets' names, you can read the paper. After editting, run the code below.

```bash
cd ./project
python main.py [-h] [--xs XS] [--ma MA] [--de DE] [--epoch EPOCH] [--lr LR]
               [--mt MT] [--bat BAT] [--tp TP] [--seed SEED] [--bign BIGN]
               [--hid HID] [--timew TIMEW]
```
               
or you can edit the train.sh, and then 

```bash
bash train.sh
```


---
### Visualization of model prediction results
![alt text](/pics/预测结果图.png)
### The model structure of plsmt-d
![alt text](/pics/模型结构图.png)

