

class DataPoint:
    def __init__(self, inputs, outputs) -> None:
        self.inputs = inputs
        self.outputs = outputs
        # how to make data ?


class NeuralNetwork:
    def __init__(self, layer_sizes) -> None:
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
    
    def compute_outputs(self, inputs):
        for layer in self.layers:
            inputs = layer.compute_outputs(inputs)
        return inputs
        
    def cost(self, data_point):
        outputs = self.compute_outputs(data_point.inputs)
        output_layer = self.layers[-1]
        cost = 0
        for node_out in range(len(outputs)-1):
            cost += output_layer.node_cost(outputs[node_out], data_point.outputs[node_out])
        return cost
    
    def global_cost(self, data):
        total_cost = 0
        for data_point in data:
            total_cost += self.cost(data_point)
        return total_cost/(len(data))
    

class Layer:
    def __init__(self, nodes_in, nodes_out) -> None:
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.weights = [] # find out how these values should be set
        self.biases = [] # find out how these values should be set
    
    def activation_function(self, weighted_input):
        return 1 if (weighted_input > 0) else 0
    
    def compute_outputs(self, inputs):
        activation_values = []
        for node_out in range(0, self.nodes_out):
            weighted_input = self.biases[node_out]
            for node_in in range(0, self.nodes_in):
                weighted_input += inputs[node_in]*self.weights[node_in][node_out]
            activation_values.append(self.activation_function(weighted_input))
        return activation_values
    
    def node_cost(self, activation_output, expected_output):
        costs = []
        for output in range(len(activation_output)-1):
            costs.append((activation_output[output]-expected_output[output])**2)
        return costs
            

    