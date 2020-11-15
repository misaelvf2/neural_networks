from models.multilayernn import MultiLayerNN
import preprocessing

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
#
# df = preprocessing.process_soybean_data()
# data = df.iloc[:, 1:-1].to_numpy().T
# raw_data = df
# labels = df.iloc[:, -1:].to_numpy().T
# classes = sorted(df['class'].unique())
# hidden_layers = 2
# num_nodes = [data.shape[0], 9, 5, len(classes)]
# learning_rate = 0.3
# epochs = 3000
# my_model = MultiLayerNN(data, raw_data, labels, classes, hidden_layers, num_nodes, learning_rate, epochs)
# my_model.multi_train()

df = preprocessing.process_forest_fires()
data = df.iloc[:, 1:-1].to_numpy().T
raw_data = df
labels = df.iloc[:, -1:].to_numpy().T
hidden_layers = 2
num_nodes = [data.shape[0], 15, 8, 1] # 10, 20, 1 for machine!
learning_rate = 0.003 # 0.001 for machine!!!
epochs = 10000 # 10000 for machine!
my_model = MultiLayerNN(data, raw_data, labels, None, hidden_layers, num_nodes, learning_rate, epochs)
my_model.regression()
