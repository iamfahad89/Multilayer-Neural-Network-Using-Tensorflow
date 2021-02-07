# Rahaman, Fahad Ur
# 1001-753-107
# 2020_10_25
# Assignment-03-01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.multinn = []
        self.weights = []
        self.dimension=input_dimension
        
    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        self.no_of_nodes=num_nodes
        
        if not self.multinn:
            matrix_of_weights=np.random.randn(self.dimension,self.no_of_nodes)
          
        else:
            matrix_of_weights=np.random.randn(self.multinn[-1]['w'].shape[1],self.no_of_nodes)
            
        generated_bias=np.random.randn(num_nodes)

        multilayer={
            'w':matrix_of_weights,
            'activation':transfer_function,
            'b':generated_bias
            }
        
        self.weights.append(None)
        self.multinn.append(multilayer)
        return None
    
    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        get_weights_without_baises = self.multinn[layer_number]["w"]
        return get_weights_without_baises

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        get_biases = self.multinn[layer_number]['b']
        return get_biases
     
    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        
        self.multinn[layer_number]['w']=weights
        return None
         
    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.multinn[layer_number]['b']=biases
        return None
    
    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        calulated_loss = [tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_hat, name=None))]
        return calulated_loss
    
    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        array_of_input=X
        f=0
        
        while(f<(len(self.multinn))):
            
            array_of_input=tf.matmul(array_of_input,self.multinn[f]['w']) 
            array_of_input=tf.add(array_of_input,self.multinn[f]['b'])

            if self.multinn[f]['activation'].lower()=='linear':
                array_of_input = array_of_input
                
            elif self.multinn[f]['activation'].lower()=='sigmoid':
                array_of_input=tf.nn.sigmoid(array_of_input)

            elif self.multinn[f]['activation'].lower()=='relu':
                array_of_input=tf.nn.relu(array_of_input)
            
            f=f+1
            
        return (array_of_input)
    
    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        p=0
         
        while(p<(num_epochs)):
             
            p=p+1
            
            for q in range(0, len(X_train), batch_size):
                batch_of_inputs = X_train[q:q+batch_size]
                batch_of_outputs = y_train[q:q+batch_size]
                W = []
                B = []
                
                for r in range (len(self.multinn)):
                    W.append(self.multinn[r]['w'])
                    B.append(self.multinn[r]['b'])
                                    
                with tf.GradientTape() as g:
                    y_hat = self.predict(batch_of_inputs)
                    calculated_loss = self.calculate_loss(batch_of_outputs, y_hat)
                    diff_loss_diff_w, diff_loss_diff_b = g.gradient(calculated_loss, [W, B])
                    
                for s in range(len(W)):
                    W[s].assign_sub(alpha * diff_loss_diff_w[s])
                    B[s].assign_sub(alpha * diff_loss_diff_b[s])
        return None

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        token = 0
        true_output = y
        actual_output = self.predict(X)
        actual_output = np.argmax(actual_output, axis =1)
        a=0

        while(a<(len(actual_output))):
            if true_output[a] != actual_output[a]:
                token = token + 1
                
            a=a+1
        percent_error = token/len(actual_output)
        return percent_error

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        true_output = y
        actual_output = self.predict(X)
        actual_output = np.argmax(actual_output,axis=1)
        confusion_matrix = [tf.math.confusion_matrix(true_output,actual_output)]
        return confusion_matrix
