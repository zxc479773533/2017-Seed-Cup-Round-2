#!/bin/bash

sed 's/\t/,/g' 'Requirements/user_info.txt' > 'src/user_info.csv'
sed 's/\t/,/g' 'Requirements/product_info.txt' > 'src/product_info.csv'
sed 's/\t/,/g' 'Requirements/behavior_info.txt' > 'src/behavior_info.csv'

(
    cd src/Empirical_analysis/
    python3 getdata.py
    python3 E_predict.py
)

(
    cd src/Xgboost_analysis/
    python3 X_predict.py
)