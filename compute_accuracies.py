import numpy as np
import pandas as pd
import os

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve
from gtda.plotting import plot_diagram

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from functions.config import *

df = pd.read_excel(os.path.join(basepath, "clinical_data.xlsx"))
df

target = 1 - df["control"].values
target

filenames = ["{}.csv".format(x) for x in df["id"]]
len(filenames)

# FA = 0, GM = 1, RS = 2
def compute_accuracies(multilayer, connectivity_type, homology_dimensions):
    print(f"Multilayer = {multilayer}")
    print(f"Connectivity = {connectivity_type}")
    print(f"Homology dimensions = {homology_dimensions}")

    if multilayer:
        data = np.zeros(shape=(len(filenames), 76*2, 76*2))
    else:
        data = np.zeros(shape=(len(filenames), 76, 76, 3))

    for i, filename in enumerate(filenames):
        df = pd.read_csv(os.path.join(basepath_FA, filename), index_col=0)
        if multilayer:
            data[i,76:,:76] = df.values
            data[i,:76,76:] = df.values
        else:
            data[i, :, :, 0] = df.values
        
        df = pd.read_csv(os.path.join(basepath_GM, filename), index_col=0)
        if multilayer:
            data[i,:76,:76] = df.values
        else:
            data[i, :, :, 1] = df.values
        
        df = pd.read_csv(os.path.join(basepath_RS, filename), index_col=0)
        if multilayer:
            data[i,76:,76:] = df.values
        else:
            data[i, :, :, 2] = df.values

    data = 1 - data

    xx = []
    for i in range(data.shape[0]):
        if multilayer:
            x = data[i, :, :]
        else:
            x = data[i, :, :, connectivity_type]
        
        np.fill_diagonal(x, 0)
        xx.append(x)

    VR = VietorisRipsPersistence(metric="precomputed", homology_dimensions=homology_dimensions, n_jobs=-1)
    diagrams = VR.fit_transform(xx)

    b = BettiCurve()
    X = b.fit_transform(diagrams)
    X = X.reshape(target.shape[0], -1)

    n_rep = 10
    auc_roc_s_lr = []
    auc_roc_s_rf = []
    auc_roc_s_nn = []

    for rep in range(n_rep):
        print("")
        print("REP: {}/{}".format(rep+1, n_rep))
        skf = StratifiedKFold(n_splits=10, shuffle=True)
        preds_lr = np.zeros(data.shape[0])
        preds_rf = np.zeros(data.shape[0])
        preds_nn = np.zeros(data.shape[0])
        fold = 0
        for train_index, test_index in skf.split(X, target):
            fold += 1
            print("Fold: {}".format(fold))

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = target[train_index], target[test_index]

            scaler = StandardScaler()
            scaler.fit(X_train)

            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            y_train_ohe = np.zeros((y_train.size, 2))
            y_train_ohe[np.arange(y_train.size), y_train] = 1
            y_test_ohe = np.zeros((y_test.size, 2))
            y_test_ohe[np.arange(y_test.size), y_test] = 1

            model_lr = LogisticRegression(max_iter=300).fit(X_train, y_train)
            model_rf = RandomForestClassifier().fit(X_train, y_train)
            model_nn = MLPClassifier(hidden_layer_sizes=[200, 100, 10], random_state=1, max_iter=300, batch_size=16).fit(X_train, y_train)

            test_preds_lr = model_lr.predict_proba(X_test)
            test_preds_rf = model_rf.predict_proba(X_test)
            test_preds_nn = model_nn.predict_proba(X_test)

            test_preds_lr = test_preds_lr[:, 1]
            test_preds_rf = test_preds_rf[:, 1]
            test_preds_nn = test_preds_nn[:, 1]

            preds_lr[test_index] = test_preds_lr
            preds_rf[test_index] = test_preds_rf
            preds_nn[test_index] = test_preds_nn
            
            auc_roc_lr = roc_auc_score(y_test, test_preds_lr)
            auc_roc_rf = roc_auc_score(y_test, test_preds_rf)
            auc_roc_nn = roc_auc_score(y_test, test_preds_nn)

            print("Test AUC LR: {:.2f}".format(auc_roc_lr))
            print("Test AUC RF: {:.2f}".format(auc_roc_rf))
            print("Test AUC NN: {:.2f}".format(auc_roc_nn))
            
        auc_roc_lr = roc_auc_score(target, preds_lr)
        auc_roc_rf = roc_auc_score(target, preds_rf)
        auc_roc_nn = roc_auc_score(target, preds_nn)

        print("")
        print("REP {} RESULTS".format(rep + 1))
        print("AUC ROC LR: {:.4f}".format(auc_roc_lr))
        print("AUC ROC RF: {:.4f}".format(auc_roc_rf))
        print("AUC ROC NN: {:.4f}".format(auc_roc_nn))

        auc_roc_s_lr.append(auc_roc_lr)
        auc_roc_s_rf.append(auc_roc_rf)
        auc_roc_s_nn.append(auc_roc_nn)

    print("")
    print("SUMMARY")
    print("AUC ROC LR: {:.4f} +- {:.4f}".format(np.mean(auc_roc_s_lr), np.std(auc_roc_s_lr)))
    print("AUC ROC RF: {:.4f} +- {:.4f}".format(np.mean(auc_roc_s_rf), np.std(auc_roc_s_rf)))
    print("AUC ROC NN: {:.4f} +- {:.4f}".format(np.mean(auc_roc_s_nn), np.std(auc_roc_s_nn)))

    return auc_roc_s_lr, auc_roc_s_rf, auc_roc_s_nn

# prepare DF to store results
df = pd.DataFrame(index=[f"rep_{i}" for i in range(1, 11)])

### Multi-layer
# compute AUC-ROC using the multi-layer and LR, RF and NN
auc_roc_s_lr_ml_h0, auc_roc_s_rf_ml_h0, auc_roc_s_nn_ml_h0 = compute_accuracies(multilayer=True, connectivity_type=None, homology_dimensions=[0])
auc_roc_s_lr_ml_h01, auc_roc_s_rf_ml_h01, auc_roc_s_nn_ml_h01 = compute_accuracies(multilayer=True, connectivity_type=None, homology_dimensions=[0, 1])
auc_roc_s_lr_ml_h012, auc_roc_s_rf_ml_h012, auc_roc_s_nn_ml_h012 = compute_accuracies(multilayer=True, connectivity_type=None, homology_dimensions=[0, 1, 2])

df["LR_ML_H0"] = auc_roc_s_lr_ml_h0
df["RF_ML_H0"] = auc_roc_s_rf_ml_h0
df["NN_ML_H0"] = auc_roc_s_nn_ml_h0

df["LR_ML_H01"] = auc_roc_s_lr_ml_h01
df["RF_ML_H01"] = auc_roc_s_rf_ml_h01
df["NN_ML_H01"] = auc_roc_s_nn_ml_h01

df["LR_ML_H012"] = auc_roc_s_lr_ml_h012
df["RF_ML_H012"] = auc_roc_s_rf_ml_h012
df["NN_ML_H012"] = auc_roc_s_nn_ml_h012

df_processed = df.transpose()
df_processed["mean"] = df_processed[[f"rep_{i}" for i in range(1, 11)]].mean(axis=1)
df_processed["std"] = df_processed[[f"rep_{i}" for i in range(1, 11)]].std(axis=1)
df_processed.to_csv(os.path.join(output_basepath, "accuracies_after_ml.csv"), float_format='%.3f', index=True, header=True)

auc_roc_s_lr_fa_h0, auc_roc_s_rf_fa_h0, auc_roc_s_nn_fa_h0 = compute_accuracies(multilayer=False, connectivity_type=0, homology_dimensions=[0])
auc_roc_s_lr_fa_h01, auc_roc_s_rf_fa_h01, auc_roc_s_nn_fa_h01 = compute_accuracies(multilayer=False, connectivity_type=0, homology_dimensions=[0, 1])
auc_roc_s_lr_fa_h012, auc_roc_s_rf_fa_h012, auc_roc_s_nn_fa_h012 = compute_accuracies(multilayer=False, connectivity_type=0, homology_dimensions=[0, 1, 2])

df["LR_FA_H0"] = auc_roc_s_lr_fa_h0
df["RF_FA_H0"] = auc_roc_s_rf_fa_h0
df["NN_FA_H0"] = auc_roc_s_nn_fa_h0

df["LR_FA_H01"] = auc_roc_s_lr_fa_h01
df["RF_FA_H01"] = auc_roc_s_rf_fa_h01
df["NN_FA_H01"] = auc_roc_s_nn_fa_h01

df["LR_FA_H012"] = auc_roc_s_lr_fa_h012
df["RF_FA_H012"] = auc_roc_s_rf_fa_h012
df["NN_FA_H012"] = auc_roc_s_nn_fa_h012

df_processed = df.transpose()
df_processed["mean"] = df_processed[[f"rep_{i}" for i in range(1, 11)]].mean(axis=1)
df_processed["std"] = df_processed[[f"rep_{i}" for i in range(1, 11)]].std(axis=1)
df_processed.to_csv(os.path.join(output_basepath, "accuracies_after_fa.csv"), float_format='%.3f', index=True, header=True)

auc_roc_s_lr_gm_h0, auc_roc_s_rf_gm_h0, auc_roc_s_nn_gm_h0 = compute_accuracies(multilayer=False, connectivity_type=1, homology_dimensions=[0])
auc_roc_s_lr_gm_h01, auc_roc_s_rf_gm_h01, auc_roc_s_nn_gm_h01 = compute_accuracies(multilayer=False, connectivity_type=1, homology_dimensions=[0, 1])
auc_roc_s_lr_gm_h012, auc_roc_s_rf_gm_h012, auc_roc_s_nn_gm_h012 = compute_accuracies(multilayer=False, connectivity_type=1, homology_dimensions=[0, 1, 2])

df["LR_GM_H0"] = auc_roc_s_lr_gm_h0
df["RF_GM_H0"] = auc_roc_s_rf_gm_h0
df["NN_GM_H0"] = auc_roc_s_nn_gm_h0

df["LR_GM_H01"] = auc_roc_s_lr_gm_h01
df["RF_GM_H01"] = auc_roc_s_rf_gm_h01
df["NN_GM_H01"] = auc_roc_s_nn_gm_h01

df["LR_GM_H012"] = auc_roc_s_lr_gm_h012
df["RF_GM_H012"] = auc_roc_s_rf_gm_h012
df["NN_GM_H012"] = auc_roc_s_nn_gm_h012

df_processed = df.transpose()
df_processed["mean"] = df_processed[[f"rep_{i}" for i in range(1, 11)]].mean(axis=1)
df_processed["std"] = df_processed[[f"rep_{i}" for i in range(1, 11)]].std(axis=1)
df_processed.to_csv(os.path.join(output_basepath, "accuracies_after_gm.csv"), float_format='%.3f', index=True, header=True)

auc_roc_s_lr_rs_h0, auc_roc_s_rf_rs_h0, auc_roc_s_nn_rs_h0 = compute_accuracies(multilayer=False, connectivity_type=2, homology_dimensions=[0])
auc_roc_s_lr_rs_h01, auc_roc_s_rf_rs_h01, auc_roc_s_nn_rs_h01 = compute_accuracies(multilayer=False, connectivity_type=2, homology_dimensions=[0, 1])
auc_roc_s_lr_rs_h012, auc_roc_s_rf_rs_h012, auc_roc_s_nn_rs_h012 = compute_accuracies(multilayer=False, connectivity_type=2, homology_dimensions=[0, 1, 2])

df["LR_RS_H0"] = auc_roc_s_lr_rs_h0
df["RF_RS_H0"] = auc_roc_s_rf_rs_h0
df["NN_RS_H0"] = auc_roc_s_nn_rs_h0

df["LR_RS_H01"] = auc_roc_s_lr_rs_h01
df["RF_RS_H01"] = auc_roc_s_rf_rs_h01
df["NN_RS_H01"] = auc_roc_s_nn_rs_h01

df["LR_RS_H012"] = auc_roc_s_lr_rs_h012
df["RF_RS_H012"] = auc_roc_s_rf_rs_h012
df["NN_RS_H012"] = auc_roc_s_nn_rs_h012

df_processed = df.transpose()
df_processed["mean"] = df_processed[[f"rep_{i}" for i in range(1, 11)]].mean(axis=1)
df_processed["std"] = df_processed[[f"rep_{i}" for i in range(1, 11)]].std(axis=1)
df_processed.to_csv(os.path.join(output_basepath, "accuracies_after_rs.csv"), float_format='%.3f', index=True, header=True)

df = df.transpose()
df["mean"] = df[[f"rep_{i}" for i in range(1, 11)]].mean(axis=1)
df["std"] = df[[f"rep_{i}" for i in range(1, 11)]].std(axis=1)
df.to_csv(os.path.join(output_basepath, "accuracies.csv"), float_format='%.3f', index=True, header=True)
