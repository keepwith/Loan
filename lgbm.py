import pandas as pd
import datetime
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
df = pd.read_csv('data/data_withfeatures.csv', )
X = df.drop('Status', axis=1)
y = df['Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# # 构造标签信息
# train_df = pd.merge(train_count_feaure, train[['user_id', 'cate', 'shop_id', 'label']].drop_duplicates(),
#                     on=['user_id', 'cate', 'shop_id'], how='left', copy=False)
# del train_count_feaure
# user_feature = ['user_id', 'age', 'sex', 'user_lv_cd', 'city_level', 'province', 'city', 'county']
# shop_feature = ['vender_id', 'shop_id', 'fans_num', 'vip_num', 'shop_main_cate', 'shop_score']
# # 用户基础特征
# train_user_f = train[user_feature].drop_duplicates()
# train_df = pd.merge(train_df, train_user_f, on=['user_id'], how='left', copy=False)
# del train_user_f
# # 店铺基础特征
# train_shop_f = train[shop_feature].drop_duplicates()
# train_df = pd.merge(train_df, train_shop_f, on=['shop_id'], how='left', copy=False)
# del train_shop_f
# # 查看样本比例
# del train
# print(train_df.groupby('label').size())

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 2 ** 5 - 1,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.7,
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_weight': 0,
    'scale_pos_weight': 25,
    'seed': 101,
    # 'nthread': 4,
    'verbose': 3,
    'use_two_round_loading': True
}
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=101)
# oof = train_df[['user_id', 'cate', 'shop_id', 'label']]
# oof['predict'] = 0
features = X_train.columns
print(features)
feature_importance_df = pd.DataFrame()
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
print(y_train)
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    # print(trn_idx)
    # print(val_idx)
    print('clf_{}'.format(fold))
    Xtrain, ytrain = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
    Xvalid, yvalid = X_train.iloc[val_idx], y_train.iloc[val_idx]
    trn_data = lgb.Dataset(Xtrain, label=ytrain)
    val_data = lgb.Dataset(Xvalid, label=yvalid)
    evals_result = {}
    lgb_clf = lgb.train(params,
                        trn_data,
                        100000,
                        valid_sets=[trn_data, val_data],
                        early_stopping_rounds=100,
                        verbose_eval=20,
                        evals_result=evals_result)
    p_valid = lgb_clf.predict(Xvalid)
    # oof['predict'][val_idx] = p_valid
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
# oof.to_csv('cache/oof.csv', index=False)
feature_importance_df.to_csv('data/feature.csv', index=False)

lgb_clf.save_model('data/model.txt')
bst = lgb.Booster(model_file='data/model.txt')
predictions = bst.predict(X_test, num_iteration=bst.best_iteration)
print(confusion_matrix(y_test, predictions))
predictions.to_csv('result/submit.csv', index=False)
model_predict_probs = bst.predict_proba(X_test)[:, 1]
lr_fpr, lr_tpr, lr_thr = roc_curve(y_test, model_predict_probs)
lr_roc_auc = auc(lr_fpr, lr_tpr)
plt.figure()
plt.plot(lr_fpr, lr_tpr, color='darkorange', lw=2, label='xgboost (area = %0.2f)' % lr_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
