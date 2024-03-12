import numpy as np
import pandas as pd

transaction_df = pd.read_csv('/home/z/PycharmProjects/model_IEEE/data/ieee-fraud-detection/train_transaction.csv')
identity_df = pd.read_csv('/home/z/PycharmProjects/model_IEEE/data/ieee-fraud-detection/train_identity.csv')
test_transaction = pd.read_csv('/home/z/PycharmProjects/model_IEEE/data/ieee-fraud-detection/test_transaction.csv')
test_identity = pd.read_csv('/home/z/PycharmProjects/model_IEEE/data/ieee-fraud-detection/test_identity.csv')

id_cols = ['card1','card2','card3','card4','card5','card6','ProductCD','addr1','addr2','P_emaildomain','R_emaildomain']
cat_cols = ['M1','M2','M3','M4','M5','M6','M7','M8','M9']
train_data_ratio = 0.8

n_train = int(transaction_df.shape[0]*train_data_ratio)
test_ids = transaction_df.TransactionID.values[n_train:]

get_fraud_frac = lambda series: 100 * sum(series)/len(series)
print("Percent fraud for train transactions: {}".format(get_fraud_frac(transaction_df.isFraud[:n_train])))
print("Percent fraud for test transactions: {}".format(get_fraud_frac(transaction_df.isFraud[n_train:])))
print("Percent fraud for all transactions: {}".format(get_fraud_frac(transaction_df.isFraud)))



with open('/home/z/PycharmProjects/model_IEEE/data/ieee-fraud-detection/test.csv', 'w') as f:
    f.writelines(map(lambda x: str(x) + "\n", test_ids))



non_feature_cols = ['isFraud', 'TransactionDT'] + id_cols
print(non_feature_cols)

feature_cols = [col for col in transaction_df.columns if col not in non_feature_cols]
print(feature_cols)