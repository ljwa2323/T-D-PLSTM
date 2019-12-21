#!/bin/bash
cd /home/ljw/project1/utils
for i in 3 4 5 6 8 9 10
do
  echo "${i} is start!"
  python generate_table.py -x ../data/data$i/Xs_gen.csv -m ../data/data$i/mask.csv -d ../data/data$i/deltat.csv
  echo "${i} is done!"
  echo "-------------"
done
