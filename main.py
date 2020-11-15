from models.multilayernn import MultiLayerNN
import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import numpy as np

# df = preprocessing.process_breast_cancer_data()
#
# data = df.iloc[:, 1:-1].to_numpy().T
# raw_data = df
# labels = df.iloc[:, -1:].to_numpy().T
# classes = [0, 1]
# hidden_layers = 2
# num_nodes = [data.shape[0], 6, 3, 1]
# learning_rate = 1.5
# epochs = 250
# my_model = MultiLayerNN(data, raw_data, labels, classes, hidden_layers, num_nodes, learning_rate, epochs)
#
# my_model.train()
# print(my_model.get_error())

# df = preprocessing.process_glass_data()
# data = df.iloc[:, 1:-1].to_numpy().T
# raw_data = df
# labels = df.iloc[:, -1:].to_numpy().T
# classes = sorted(df['class'].unique())
# hidden_layers = 2
# num_nodes = [data.shape[0], 9, 5, len(classes)]
# learning_rate = 0.3
# epochs = 1000
# my_model = MultiLayerNN(data, raw_data, labels, classes, hidden_layers, num_nodes, learning_rate, epochs)
# my_model.multi_train()

# df = preprocessing.process_forest_fires()
# data = df.iloc[:, 1:-1].to_numpy().T
# raw_data = df
# labels = df.iloc[:, -1:].to_numpy().T
# hidden_layers = 2
# num_nodes = [data.shape[0], 20, 20, 1] # 10, 20, 1 for machine! 15, 20 for forest fires
# learning_rate = 0.01 # 0.001 for machine!!!, 0.006 for forest fires
# epochs = 10000 # 10000 for machine!
# my_model = MultiLayerNN(data, raw_data, labels, None, hidden_layers, num_nodes, learning_rate, epochs)
# my_model.regression()

def main(dataset, hidden_layers, num_nodes, learning_rate, epochs):
    if dataset == 'breast':
        df = preprocessing.process_breast_cancer_data()
        modality = 'binary'
    elif dataset == 'glass':
        df = preprocessing.process_glass_data()
        modality = 'multi'
    elif dataset == 'soybean':
        df = preprocessing.process_soybean_data()
        modality = 'multi'
    elif dataset == 'abalone':
        df = preprocessing.process_abalone()
        modality = 'regression'
    elif dataset == 'machine':
        df = preprocessing.process_machine()
        modality = 'regression'
    elif dataset == 'forest_fires':
        df = preprocessing.process_forest_fires()
        modality = 'regression'
    else:
        df = preprocessing.process_breast_cancer_data()
        modality = 'binary'

    if modality == 'regression':
        skf = KFold(n_splits=5, shuffle=True, random_state=5)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    training_sets, test_sets = [], []
    for fold, (train, test) in enumerate(skf.split(X=np.zeros(len(df)), y=df.iloc[:, -1:])):
        training_sets.append(df.iloc[train])
        test_sets.append(df.iloc[test])

    # Train; run 5 experiments in total
    training_errors, trained_models = [], []
    for training_set in training_sets:
        print("\nTraining:")
        training_data = training_set.iloc[:, 1:-1].to_numpy().T
        training_labels = training_set.iloc[:, -1:].to_numpy().T
        if modality == 'regression':
            classes = None
            actual_num_nodes = [training_data.shape[0]] + num_nodes + [1]
        else:
            classes = sorted(df['class'].unique())
            actual_num_nodes = [training_data.shape[0]] + num_nodes + [len(classes)]
        my_model = MultiLayerNN(training_data, training_set, training_labels, classes, hidden_layers, actual_num_nodes, learning_rate, epochs)
        if modality == 'binary':
            my_model.train()
        elif modality == 'multi':
            my_model.multi_train()
        elif modality == 'regression':
            my_model.regression()
        trained_models.append(my_model)
        if modality == 'regression':
            training_errors.append(my_model.get_training_mse())
        else:
            training_errors.append(my_model.get_training_error())
        my_model.plot_error()
        # my_model.report_classifications()

    # Test; run 5 experiments in total
    # testing_errors = []
    # for model, test_set in zip(trained_models, test_sets):
    #     print("\nTesting: ")
    #     testing_data = test_set.iloc[:, 1:-1].to_numpy().T
    #     testing_labels = test_set.iloc[:, -1:].to_numpy().T
    #     if modality == 'binary':
    #         model.test(testing_data, testing_labels)
    #     elif modality == 'multi':
    #         model.multi_test(testing_data, testing_labels)
    #     elif modality == 'regression':
    #         model.multi_regression()
    #     testing_errors.append(model.get_testing_error())
        # model.report_classifications()

    # Report average results
    # average_training_error = sum(training_errors) / len(training_errors)
    # average_testing_error = sum(testing_errors) / len(testing_errors)
    # print("\nSummary:")
    # print(f"Average training error: {average_training_error}")
    # print(f"Average testing error: {average_testing_error}")


main('machine', 2, [20, 15], 0.001, 5000)
