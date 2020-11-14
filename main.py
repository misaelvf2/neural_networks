from models.multilayernn import MultiLayerNN
import preprocessing

df = preprocessing.process_breast_cancer_data()

data = df.iloc[:, 1:-1].to_numpy().T
labels = df.iloc[:, -1:].to_numpy().T
classes = [0, 1]
hidden_layers = 2
num_nodes = [data.shape[0], 6, 3, 1]
learning_rate = 1.5
epochs = 250
my_model = MultiLayerNN(data, labels, classes, hidden_layers, num_nodes, learning_rate, epochs)

my_model.train()
print(my_model.get_error())
