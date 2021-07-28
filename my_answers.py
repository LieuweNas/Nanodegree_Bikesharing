import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### Set self.activation_function to sigmoid function
        self.activation_function = lambda x: (1 / (1 + np.exp(-x)))

        self.activation_function_derivative = lambda x: (
                    self.activation_function(x) * (1 - self.activation_function(x)))
              
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####

        ### Forward pass ###

        # Input to hidden layer output
        hidden_inputs = np.matmul(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Hidden layer output to NN output
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        # Calculate error
        output_error = final_outputs - y
        output_error_delta = output_error * 1 # Derivative of f(x) = x is 1 so error delta is

        # Calculate the hidden layer's contribution to the error via backpropagation
        hidden_layer_error = np.matmul(output_error_delta[:, None], self.weights_hidden_to_output.T)
        test_derivative_calc = self.activation_function_derivative(hidden_outputs[:, None])
        hidden_layer_error_delta = hidden_layer_error.T * test_derivative_calc

        # Weight step (hidden to output)
        hidden_outputs_transform = hidden_outputs[:,None]
        delta_weights_h_o += np.matmul(hidden_outputs_transform, output_error_delta[:, None])

        # Weight step (input to hidden)
        X_transform = X[:, None]
        delta_weights_i_h += np.matmul(X_transform, hidden_layer_error_delta.T)

        #print(f'delta_weights_i_h: {delta_weights_i_h}')
        
        return delta_weights_i_h, delta_weights_h_o
        

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output -= (((delta_weights_h_o) / n_records) * self.lr)
        self.weights_input_to_hidden -= (((delta_weights_i_h) / n_records) * self.lr)

        #print(f'updating self.weights_input_to_hidden: {self.weights_input_to_hidden.shape}')
        

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # Input to hidden layer output
        hidden_inputs = np.matmul(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Hidden layer output to NN output
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 2000
learning_rate = 0.1
hidden_nodes = 30
output_nodes = 1
