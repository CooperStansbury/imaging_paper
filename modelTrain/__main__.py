import pandas as pd
import numpy as np
import os
import json
import shutil
import json
import argparse
from xml.dom import minidom
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.metrics as skmet
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load


def load_config(config_path):
    """Load json with user params """
    with open(config_path) as json_file:
        data = json.load(json_file)
    return data

    
def col_renamer(df, str_match, new_name):
    """A function to rename columns with a substrig substitution """
    old_names = [x for x in df.columns if str_match in x]
    new_names = [x.replace(str_match, new_name) for x in old_names]
    rename_dict = dict(zip(old_names, new_names))
    
    df = df.rename(columns=rename_dict)
    return df
    
    
def load_trackmate_path(trackmate_path):
    """A function to load FILTERED spots """
    return pd.read_csv(trackmate_path)


def load_cvat_xml(fpath):
    """A function to load annotations from CVAT into a pandas dataframe"""
    new_rows = []
    f = open(fpath,'r')
    xmldoc = minidom.parse(f)
    items = xmldoc.getElementsByTagName('image')
    
    for item in items:
        try:
            l = item.getElementsByTagName('tag')
            assert(len(l) > 0)
        except:
            continue
        
        FILENAME = item.attributes['name'].value
        TRACK_ID = FILENAME.split("_frame_")[0]
        FRAME = int(FILENAME.split("frame_")[1].replace(".png", ""))
        LABEL = item.getElementsByTagName('tag')[0].attributes['label'].value

        row = {
            "id" : item.attributes['id'].value,
            "filename" :FILENAME,
            "FRAME" : FRAME,
            "TRACK_ID" : TRACK_ID,
            "category_id" : LABEL
        }
        new_rows.append(row)
        
    return pd.DataFrame(new_rows)
    
    
def one_hot_encode(df):
    """A function to one-hot encode annotations labels """
    one_hot = pd.get_dummies(df['category_id'])
    df = df.join(one_hot)
    return df
    
def handle_missing(df, x_columns):
    """A function to replace missing values with the max of the column """
    pd.options.mode.use_inf_as_na = True
    for c in x_columns:
        if df[c].isna().sum() > 0:
            df[c] = np.where(df[c].isna(), df[c].max(), df[c])
    return df


def make_val_train_test(df, x_columns, verbose=False):
    """A function to make trainning and valdation columns"""
    y_columns = ['G1', 'S', 'G2', 'NA']
    X, X_val, y, y_val = train_test_split(df[x_columns], df[y_columns], test_size=0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    if verbose:
        print(f"\n{X_train.shape=}")
        print(f"{y_train.shape=}")

        print(f"\n{X_test.shape=}")
        print(f"{y_test.shape=}")

        print(f"\n{X_val.shape=}")
        print(f"{y_val.shape=}")
    
    return X_val, y_val, X_train, y_train, X_test, y_test
    

def preprocess(X_val, X_train, X_test, x_columns):
    """A function to fit a scaler and transform the val and test
    sets by that scaler """
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_train.columns = x_columns
    
    # use the trainning data to transform the testing data
    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test)
    X_test.columns = x_columns

    X_val = scaler.transform(X_val)
    X_val = pd.DataFrame(X_val)
    X_val.columns = x_columns
    
    return X_val, X_train, X_test, scaler

    
def get_diagnostoics(model, X_test, y_test, outpath, verbose=True):
    y_pred = model.predict(X_test)
    auc = skmet.roc_auc_score(y_test, y_pred, multi_class='ovo')
    print(f"{auc=:.5f}")
    print()

    target_names = ['G1', 'S', 'G2/M', 'NA']
    cm = skmet.confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1))
    cm = pd.DataFrame(cm)
    cm.index = target_names
    cm.columns = target_names
    confusion_path = f"{outpath}confusion_matrix.csv"
    cm.to_csv(confusion_path)
    print(f"saved: {confusion_path}")
    
    if verbose:
        print(cm)

    report = skmet.classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    report = pd.DataFrame(report).transpose()
    report_path = f"{outpath}classification_report.csv"
    report.to_csv(report_path)
    print(f"saved: {report_path}")
    
    if verbose:
        print(report)
         
    feature_importances = pd.DataFrame(model.feature_importances_,
                                   index = X_train.columns,
                                   columns=['importance']).sort_values('importance',ascending=False)

    feature_importances['cumsum'] = feature_importances['importance'].cumsum()
    importance_path = f"{outpath}feature_importances.csv"
    feature_importances.to_csv(importance_path)
    print(f"saved: {importance_path}")

    
def namestr(obj, namespace):
    """ get the name of the variable """
    return [name for name in namespace if namespace[name] is obj]   
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", nargs='?', default='modelTrain/config.json', 
                        help="The path to a processed alignment table.")
    args = parser.parse_args()
    
    # argument parsing 
    config_path = args.config
    config = load_config(config_path)
    # load all config params as variables
    # these are all UPPERCASE
    locals().update(config)
    
    # load the filtered trackmate file
    df = load_trackmate_path(TRACKMATE_PATH)
    
    # load the annotations
    annotations = load_cvat_xml(ANNOTATION_PATH)
    
    # merge annotations
    df = pd.merge(
        df, annotations, 
        on=['TRACK_ID', 'FRAME'],
        how='left'
    )
    
    # clean up
    del annotations
    
    # rename some columns for easier debugging later
    df = col_renamer(df, 'CH1', 'RED')
    df = col_renamer(df, 'CH2', 'GREEN')
    df = col_renamer(df, 'CH3', 'BLUE')
    
    # encoding and missing data handling
    df = one_hot_encode(df)
    df = handle_missing(df, TRAINNING_COLS)
    
    # train/test/val split
    X_val, y_val, X_train, y_train, X_test, y_test = make_val_train_test(df, TRAINNING_COLS)
    
    # scale the data
    X_val, X_train, X_test, scaler = preprocess(X_val, X_train, X_test, TRAINNING_COLS)
    
    # fit the model
    model = RandomForestClassifier(bootstrap=True,
                               n_estimators=100)

    model.fit(X_train, y_train)
    
    model_path = f"{MODEL_OUTPUTS}/model.joblib"
    scaler_path = f"{MODEL_OUTPUTS}/scaler.joblib"
    
    # save the model and scaler
    dump(model, model_path)
    dump(scaler, scaler_path)
    
    get_diagnostoics(model, X_test, y_test, MODEL_OUTPUTS)
    
    datasets = [X_val, y_val, X_train, y_train, X_test, y_test]
    
    for d in datasets:
        outpath = f"{MODEL_OUTPUTS}{namestr(d, globals())[0]}.csv"
        d.to_csv(outpath, index=False)
        print(f"saved: {outpath}")
        
    print("done.")
    