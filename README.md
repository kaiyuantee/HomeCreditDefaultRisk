# Home Credit Default Risk

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

## Overview

- This model uses LightGBM with goss and label encoding for the application's categorical features. 
- Other tables are using one-hot encode with mean, sum and a few different functions to aggregate. 
- The main idea was to add more time related features like last application and last X months aggregations.
- There are also aggregations for specific loan types and status as well as ratios between tables. Configurations are in src/cfgs.py file.

## Requirements 
```
pip install -r requirements.txt
python setup.py develop
```

## Run
```
python -m src/main.py
```

