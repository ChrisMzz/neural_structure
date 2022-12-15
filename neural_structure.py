import numpy as np

class DataPoint:
    def __init__(self, inputs, outputs) -> None:
        self.inputs = inputs
        self.outputs = outputs
        # how to make data ?

    def __str__(self):
        return f"{self.inputs}\n{self.outputs}"


class NeuralNetwork:
    def __init__(self, layer_sizes) -> None:
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
    
    def compute_outputs(self, inputs):
        for layer in self.layers:
            inputs = layer.compute_outputs(inputs)
        return inputs

    def classify(self, inputs):
        outputs = self.compute_outputs(inputs)
        max_value = max(outputs)
        return outputs.index(max_value)
    
        
    def cost_on_data_point(self, data_point):
        outputs = self.compute_outputs(data_point.inputs)
        output_layer = self.layers[-1]
        cost = 0
        for node_out in range(len(outputs)-1):
            cost += output_layer.node_cost(outputs[node_out], data_point.outputs[node_out])
        return cost
    
    def cost(self, data):
        total_cost = 0
        for data_point in data:
            total_cost += self.cost_on_data_point(data_point)
        return total_cost/(len(data))

    def apply_all_gradients(self, learn_rate):
        for layer in self.layers:
            layer.apply_gradients(learn_rate)

    def learn(self, training_data, learn_rate):
        h = 0.01
        original_cost = self.cost(training_data)
        for layer in self.layers:
            for node_in in range(layer.nodes_in):
                for node_out in range(layer.nodes_out):
                    layer.weights[node_in][node_out] += h
                    delta_cost = self.cost(training_data) - original_cost
                    layer.weights[node_in][node_out] -= h
                    layer.cost_gradient_W[node_in][node_out] = delta_cost/h
            for bias in range(len(layer.biases)-1):
                layer.biases[bias] += h
                delta_cost = self.cost(training_data) - original_cost
                layer.biases[bias] -= h
                layer.cost_gradient_B[bias] = delta_cost/h
        self.apply_all_gradients(learn_rate)


class Layer:
    def __init__(self, nodes_in, nodes_out) -> None:
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.weights = [[0 for _ in range(self.nodes_out)] for _ in range(self.nodes_in)]
        self.biases = [0 for _ in range(self.nodes_out)]
        self.cost_gradient_W = []
        self.cost_gradient_B = []
        self.set_random_weights()
        self.activation_function = lambda weighted_input : 1 if (weighted_input > 0) else 0

    def set_af(self, function):
        self.activation_function = function
    
    def compute_outputs(self, inputs):
        activation_values = []
        for node_out in range(self.nodes_out):
            weighted_input = self.biases[node_out]
            for node_in in range(self.nodes_in):
                weighted_input += inputs[node_in]*self.weights[node_in][node_out]
            activation_values.append(self.activation_function(weighted_input))
        return activation_values
    
    def node_cost(self, activation_output, expected_output):
        costs = []
        for output in range(len(activation_output)-1):
            costs.append((activation_output[output]-expected_output[output])**2)
        return costs
            
    def apply_gradients(self, learn_rate):
        for node_out in range(self.nodes_out):
            self.biases[node_out] -= self.cost_gradient_B[node_out]*learn_rate
            for node_in in range(self.nodes_in):
                self.weights[node_in][node_out] = self.cost_gradient_W[node_in][node_out]*learn_rate
    
    def set_random_weights(self):
        for node_in in range(self.nodes_in):
            for node_out in range(self.nodes_out):
                weight = np.random.rand()*2 - 1
                self.weights[node_in][node_out]

