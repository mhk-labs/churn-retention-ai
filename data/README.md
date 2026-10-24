# Data

Main source: https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data

Each contributor must obtain access and accept applicable competition terms.
Download locally into data/raw/; never commit downloaded data.
The data lead will add a versioned download/sampling script after verifying the files.
Record filenames, versions, sizes, date ranges, checksums and the sampling seed.
Sample customer IDs once, then filter every related table to the same users.
Use chunked reads if logs do not fit memory. Store derived tables in data/processed/.
Public deployment must use synthetic or otherwise permitted demo data.
