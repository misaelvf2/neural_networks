import pandas as pd
import numpy as np

def process_breast_cancer_data():
    """
    Imports breast cancer dataset, imputes missing values, and normalizes
    :return: DataFrame
    """
    df = pd.read_csv("data/breast-cancer-wisconsin.data", header=None, na_values=['?'])
    df = df.fillna(np.random.randint(1, 11))
    df.columns = ['id', 'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion',
                  'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
    df['class'] = df['class'].apply(lambda x: 1 if x == 4 else 0)
    normalized_df = (df.iloc[:, 1:-1] - df.iloc[:, 1:-1].mean()) / df.iloc[:, 1:-1].std()
    normalized_df.insert(0, 'id', df['id'])
    normalized_df.insert(len(df.columns) - 1, 'class', df['class'])
    normalized_df = normalized_df.sample(frac=1)
    return normalized_df


def process_glass_data():
    """
    Imports glass data, and normalizes
    :return: DataFrame
    """
    df = pd.read_csv("data/glass.data", header=None)
    df.columns = ['id', 'ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe', 'class']
    normalized_df = (df.iloc[:, 1:-1] - df.iloc[:, 1:-1].mean()) / df.iloc[:, 1:-1].std()
    normalized_df.insert(0, 'id', df['id'])
    normalized_df.insert(len(df.columns) - 1, 'class', df['class'])
    normalized_df = normalized_df.sample(frac=1)
    return normalized_df


def process_soybean_data():
    """
    Imports soybean data, and normalizes
    :return: DataFrame
    """
    df = pd.read_csv("data/soybean-small.data", header=None)
    df.columns = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged',
                  'severity', 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo',
                  'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild', 'stem',
                  'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies', 'external decay',
                  'mycelium', 'int-discolor', 'slcerotia', 'fruit-pods', 'fruit spots', 'seed',
                  'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots', 'class']
    df['class'] = df['class'].apply(convert_soybean_to_numerical)
    df = df.sample(frac=1)
    return df

def convert_soybean_to_numerical(x):
    """
    Helper function for soybean data. Converts class labels to numerical data.
    :param x: Str
    :return: Int
    """
    if x == 'D1':
        return 1
    elif x == 'D2':
        return 2
    elif x == 'D3':
        return 3
    elif x == 'D4':
        return 4


def process_abalone():
    """
    Imports Abalone dataset
    :return: DataFrame
    """
    df = pd.read_csv("data/abalone.data", header=None)
    df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                  'Shell_weight', 'class']
    df['Sex'] = df['Sex'].apply(convert_abalone_to_numerical)
    normalized_df = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / df.iloc[:, :-1].std()
    # normalized_df.insert(0, 'Sex', df['Sex'])
    normalized_df.insert(len(df.columns) - 1, 'class', df['class'])
    normalized_df = normalized_df.sample(frac=1)
    return normalized_df

def convert_abalone_to_numerical(x):
    """
    Helper function for abalone data. Converts sex labels to numerical data.
    :param x: Str
    :return: Int
    """
    if x == 'M':
        return 1
    elif x == 'F':
        return 2
    elif x == 'I':
        return 3


def process_machine():
    """
    Imports Machine dataset
    :return: DataFrame
    """
    df = pd.read_csv("data/machine.data", header=None)
    df.columns = ['Vendor', 'Id', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN',
                  'CHMAX', 'class', 'ERP']
    df = df.drop(columns=['Vendor', 'Id', 'ERP'])
    df = df[['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN',
                  'CHMAX', 'class']]
    normalized_df = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / df.iloc[:, :-1].std()
    normalized_df.insert(len(df.columns) - 1, 'class', df['class'])
    normalized_df = normalized_df.sample(frac=1)
    return normalized_df


def process_forest_fires():
    """
    Imports Forest Fires dataset
    :return: DataFrame
    """
    df = pd.read_csv("data/forestfires.csv", header=None)
    df.columns = ['X', 'Y', 'Month', 'Day', 'FFMC', 'DMC', 'DC', 'ISI', 'Temp',
                  'RH', 'Wind', 'Rain', 'class']
    df['Month'] = df['Month'].apply(change_month_to_number)
    df['Day'] = df['Day'].apply(change_day_to_number)
    normalized_df = (df.iloc[:, :-1] - df.iloc[:, :-1].mean()) / df.iloc[:, :-1].std()
    normalized_df.insert(len(df.columns) - 1, 'class', df['class'])
    normalized_df = normalized_df.sample(frac=1)
    return normalized_df


def change_month_to_number(x):
    """
    Helper function for forest fires data. Converts months labels to numerical data.
    :param x: Str
    :return: Int
    """
    if x == 'jan':
        return 1
    elif x == 'feb':
        return 2
    elif x == 'mar':
        return 3
    elif x == 'apr':
        return 4
    elif x == 'may':
        return 5
    elif x == 'jun':
        return 6
    elif x == 'jul':
        return 7
    elif x == 'aug':
        return 8
    elif x == 'sep':
        return 9
    elif x == 'oct':
        return 10
    elif x == 'nov':
        return 11
    elif x == 'dec':
        return 12


def change_day_to_number(x):
    """
    Helper function for forest fires data. Converts day labels to numerical data.
    :param x: Str
    :return: Int
    """
    if x == 'sun':
        return 1
    elif x == 'mon':
        return 2
    elif x == 'tue':
        return 3
    elif x == 'wed':
        return 4
    elif x == 'thu':
        return 5
    elif x == 'fri':
        return 6
    elif x == 'sat':
        return 7