#!/usr/bin/env python
# coding:utf-8

#%%
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy import stats
import click
import joblib
import time

np.seterr(divide='ignore',invalid='ignore')

@click.command()
@click.option("--model", default="xgb", show_default=True, help="model name")
@click.option("--test", default='feature_test.csv', show_default=True, help="model name")
@click.option("--train", default='feature_train.csv', show_default=True, help="model name")
@click.option("--out", default='log_evaluation.txt', show_default=True, help="define output log-file name")
@click.option("--savemod", is_flag=True, show_default=True, help="train and save SFs model")
@click.option("--runmod", is_flag=True, help="run trained SFs model, the model is named joblib.dat by default")
def main(model,train,test,out,savemod,runmod):
    global X_train,y_train,X_test,y_test,trained_model

    if runmod:
        print('# run model #')
        run_model(test,'joblib.dat')
    else:
        X_train,y_train,X_test,y_test=read_data(train,test)
        modlist={'xgb': train_xgb_model(), 'rf':train_rf_model(), 'gbdt':train_gbdt_model(), \
        'et':train_et_model()}
        if model == 'all':
            for mod in modlist:
                print(f'\n{mod}')
                trained_model = modlist[mod]
                evaluation_method(mod,out)
                save_model(savemod,model)
        else:
            trained_model = modlist[model]
            evaluation_method(model,out)
            save_model(savemod,model)


def run_model(test,out,model_dat='joblib.dat'):
    df = pd.read_csv(test,sep=',')
    num_columns=list(df.columns[1:-1])
    X_test = df[num_columns]
    y_test = df.iloc[:,-1].to_numpy()
    trained_model = joblib.load(model_dat)
    y_test_pred = trained_model.predict(X_test)
    test_pears = pearsonr(y_test, y_test_pred)[0]
    test_spearman=stats.spearmanr(y_test, y_test_pred)[0]
    test_rmse = mean_squared_error(y_test, y_test_pred)**0.5
    print('\n## test set ##')
    print(f"pears:   {test_pears:.4f}")
    print(f"spearman:{test_spearman:.4f}")
    print(f"rmse:    {test_rmse:.4f}")
    pdb = df['pdbid'].to_numpy()
    save_testdata(y_test,y_test_pred,pdb,out)


def read_data(train_data,test_data):
    df_train = pd.read_csv(train_data,sep=',')
    df_test = pd.read_csv(test_data,sep=',')    
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train.dropna(inplace=True)
    global num_columns

    num_columns=list(df_train.columns[1:-1])

    global standardScaler
    standardScaler = StandardScaler()
    
    X_train = df_train[num_columns]
    y_train = df_train.iloc[:,-1].to_numpy()

    X_test = df_test[num_columns]
    y_test = df_test.iloc[:,-1].to_numpy()
    global pdb
    pdb = df_test['pdbid'].to_numpy()
    return X_train,y_train,X_test,y_test


def train_xgb_model():
    param={'n_estimators': 7546, 'max_depth': 7, 'reg_alpha': 3, 'reg_lambda': 5, 'min_child_weight': 1, 'gamma': 0, 'learning_rate': 0.032, 'colsample_bytree': 0.67}
    param1 = {
            'booster':'gbtree',
            'objective':'reg:squarederror',
            'subsample': 0.9,

        }
    param2 = {**param,**param1}
    xlf = xgb.XGBRegressor(**param2)
    xlf.fit(X_train, y_train, eval_metric='rmse', verbose = False, eval_set = [(X_test, y_test)],early_stopping_rounds=100)
    #xlf.fit(X_train, y_train)
    return xlf

def train_rf_model():
    from sklearn.ensemble import RandomForestRegressor
    random_model = RandomForestRegressor(random_state=1206, n_estimators=500, n_jobs=8, oob_score=True, max_features=0.33)
    random_model.fit(X_train, y_train)
    return random_model

def train_et_model():
    from sklearn.ensemble import ExtraTreesRegressor
    params={'n_estimators':50,'random_state':0}
    et_model=ExtraTreesRegressor(**params)
    et_model.fit(X_train, y_train)
    return et_model

def train_svm_model():
    from sklearn.svm import SVR
    svm_model=SVR(max_iter=10000)
    svm_model.fit(X_train, y_train)
    return svm_model

def train_gbdt_model():
    from sklearn.ensemble import GradientBoostingRegressor
    regr = GradientBoostingRegressor(random_state=1206, n_estimators=20000, max_features="sqrt", max_depth=8, min_samples_split=3, learning_rate=0.005, loss="squared_error", subsample=0.7)
    regr.fit(X_train,y_train)
    return regr

def evaluation_method(model_name,out='log_evaluation.txt'):
    global y_train_pred,y_test_pred
    y_train_pred = trained_model.predict(X_train)
    y_test_pred = trained_model.predict(X_test)

    train_pears = pearsonr(y_train, y_train_pred)[0]
    train_spearman=stats.spearmanr(y_train, y_train_pred)[0]
    train_rmse = mean_squared_error(y_train, y_train_pred)**0.5
    test_pears = pearsonr(y_test, y_test_pred)[0]
    test_spearman=stats.spearmanr(y_test, y_test_pred)[0]
    test_rmse = mean_squared_error(y_test, y_test_pred)**0.5
    print('## training set ##')
    print(f"pears:   {train_pears:.4f}")
    print(f"spearman:{train_spearman:.4f}")
    print(f"rmse:    {train_rmse:.4f}")    
    print('\n## test set ##')
    print(f"pears:   {test_pears:.4f}")
    print(f"spearman:{test_spearman:.4f}")
    print(f"rmse:    {test_rmse:.4f}")

    localtime = time.asctime( time.localtime(time.time()) )
    with open(out,'a') as f:
        f.write(f'''
{localtime}
{model_name}
## training set ##
pears:   {train_pears:.4f}
spearman:{train_spearman:.4f}
rmse:    {train_rmse:.4f}   
## test set ##
pears:   {test_pears:.4f}
spearman:{test_spearman:.4f}
rmse:    {test_rmse:.4f}
        ''')


    df_test = pd.DataFrame({
                        "pdbid":list(pdb),
                        "label":list(y_test),
                        "pred":list(y_test_pred)})
    df_test.to_csv(f"{out}.predict", sep=' ', index='')#new
    print(f"Results was saved in {out}.predict")


def save_testdata(y_test,y_test_pred,pdb,out):
    df_test = pd.DataFrame({
                        "pdbid":list(pdb),
                        "label":list(y_test),
                        "pred":list(y_test_pred)})
    df_test.to_csv(f"{out}.predict", sep=' ', index='')
    print(f"Results was saved in {out}.predict")
    

def save_model(savemod,model):
    if savemod:
        import joblib
        joblib.dump(trained_model, f"joblib.dat")

if __name__ == '__main__':
    main()
# %%



