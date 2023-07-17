import nnfs 
import numpy as np
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data


nnfs.init()

class Layer:
    class Dense:
        def __init__(self,n_inputs,n_neurons,weight_regularizer_l1=0,weight_regularizer_l2=0,bias_regularizer_l1=0,bias_regularizer_l2=0):
            self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
            self.biases = np.zeros((1,n_neurons))

            self.weight_regularizer_l1 = weight_regularizer_l1
            self.weight_regularizer_l2 = weight_regularizer_l2
            self.bias_regularizer_l1 = bias_regularizer_l1
            self.bias_regularizer_l2 = bias_regularizer_l2

        def forward(self,inputs):
            self.inputs = inputs
            self.output = np.dot(inputs,self.weights) + self.biases

        def backward(self,dvalues):
            self.dweights = np.dot(self.inputs.T,dvalues)
            self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
            

            if self.weight_regularizer_l1 > 0:
                dL1 = np.ones_like(self.weights)
                dL1[self.weights<0] = -1
                self.dweights += self.weight_regularizer_l1*dL1
            
            if self.weight_regularizer_l2 > 0:
                self.dweights += 2 * self.weight_regularizer_l2 * self.weights
            
            if self.bias_regularizer_l1 > 0:
                dL1 = np.ones_like(self.biases)
                dL1[self.biases < 0] = -1
                self.dbiases += self.bias_regularizer_l1*dL1
            
            if self.bias_regularizer_l2 > 0:
                self.dbiases += 2*self.bias_regularizer_l2*self.biases

            self.dinputs = np.dot(dvalues,self.weights.T)
    class Dropout:
    
        def __init__(self,rate):
            self.rate = 1 - rate

        def forward(self,inputs,training):
            self.inputs = inputs 

            if not training:
                self.output = inputs.copy()
                return
            self.binary_mask = np.random.binomial(1,self.rate,size= inputs.shape)/self.rate 
            self.output = inputs*self.binary_mask

        def backward(self,dvalues):
            self.dinputs = dvalues*self.binary_mask
    class Input:
        def forward(self,inputs):
            self.output = inputs

class ActivationFunctions():

    """
    This class contains all the activation function classes that can be used in the neural network seperately.
    """
    class Activation:
        """
        This is the base class for all the activation functions.
        Attributes:
            output: This is the output of the activation function.

        Methods:
            forward(inputs): This method is used to calculate the output of the activation function.
        """
        def __init__(self):
            """
            This is the constructor for the base class.
            Attributes:
                output: This is the output of the activation function.
            """
            self.output = None
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.inputs = inputs
        def backward(self,dvalues):
            """
            This method is used to calculate the derivative of the activation function.
            Args:
                dvalues: This is the derivative of the loss function with respect to the output of the activation function.
            """
            self.dinputs = dvalues.copy()

    class ReLU(Activation):
        """
        This class implements the ReLU activation function.
        Attributes:
            output: This is the output of the activation function.
        Methods:
            forward(inputs): This method is used to calculate the output of the activation function.
        """
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.inputs = inputs
            self.output = np.maximum(0,inputs)
        
        def backward(self,dvalues):
            self.dinputs = dvalues.copy()
            self.dinputs[self.inputs <= 0] = 0
        
        def predictions(self,outputs):
            return outputs

    class Softmax(Activation):
        """
        This class implements the Softmax activation function.
        Attributes:
            output: This is the output of the activation function.
        Methods:
            forward(inputs): This method is used to calculate the output of the activation function.
        """
        def forward(self, inputs):
            # Remember input values
            self.inputs = inputs

            # Get unnormalized probabilities
            exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                                keepdims=True))
            # Normalize them for each sample
            probabilities = exp_values / np.sum(exp_values, axis=1,
                                                keepdims=True)

            self.output = probabilities

    # Backward pass
        def backward(self, dvalues):

            # Create uninitialized array
            self.dinputs = np.empty_like(dvalues)

            # Enumerate outputs and gradients
            for index, (single_output, single_dvalues) in \
                    enumerate(zip(self.output, dvalues)):
                # Flatten output array
                single_output = single_output.reshape(-1, 1)
                # Calculate Jacobian matrix of the output
                jacobian_matrix = np.diagflat(single_output) - \
                                np.dot(single_output, single_output.T)
                # Calculate sample-wise gradient
                # and add it to the array of sample gradients
                self.dinputs[index] = np.dot(jacobian_matrix,
                                            single_dvalues)
                
        def predictions(self,outputs):
            return np.argmax(outputs,axis=1)
    class Sigmoid(Activation):
        """
        This class implements the Sigmoid activation function.
        Attributes:
            output: This is the output of the activation function.
        Methods:
            forward(inputs): This method is used to calculate the output of the activation function.
        """
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.inputs = inputs
            self.output = 1/(1+np.exp(-inputs))
        def backward(self,dvalues):
            self.dinputs = dvalues*(1-self.output)*self.output

        def predictions(self,outputs):
            return (outputs > 0.5)*1
    class TanH(Activation):
        """
        This class implements the TanH activation function.
        Attributes:
            output: This is the output of the activation function.
        Methods:
            forward(inputs): This method is used to calculate the output of the activation function.
        """
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.tanh(inputs)

    class Linear(Activation):
        """
        This class implements the Linear activation function.
        Attributes:
            output: This is the output of the activation function.
        Methods:
            forward(inputs): This method is used to calculate the output of the activation function.
        """
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = inputs
            self.inputs = inputs
        
        def backward(self, dvalues):
            super().backward(dvalues)
        
        def predictions(self,outputs):
            return outputs

    class LeakyReLU(Activation):
        """
        This class implements the LeakyReLU activation function.
        Attributes:
            output: This is the output of the activation function.
            alpha: This is the slope of the activation function.
        Methods:
            forward(inputs): This method is used to calculate the output of the activation function.    
        """
        def __init__(self,alpha=0.01):
            """
            This is the constructor for the LeakyReLU class.
            Attributes:
                output: This is the output of the activation function.
                alpha: This is the slope of the activation function.
            """ 
            super().__init__()
            self.alpha = alpha
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.maximum(self.alpha*inputs,inputs)

    class ExpLU(Activation):
        def __init__(self,alpha=0.01):
            super().__init__()
            self.alpha = alpha
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.where(inputs<0,self.alpha*(np.exp(inputs)-1),inputs)

    class Swish(Activation):
        def __init__(self):
            super().__init__()
            self.beta = 1
            self.sigmoid = ActivationFunctions.Sigmoid()

        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = inputs*self.sigmoid.forward(self.beta*inputs)

    class Maxout(Activation):
        def __init__(self):
            super().__init__()
            self.w1 = 0.01*np.random.randn(1,1)
            self.w2 = 0.01*np.random.randn(1,1)
            self.b1 = 0
            self.b2 = 0

        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.maximum(np.dot(inputs,self.w1)+self.b1,np.dot(inputs,self.w2)+self.b2)

    class SoftPlus(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.log(1+np.exp(inputs))

    class BentIdentity(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = ((np.sqrt(np.square(inputs)+1)-1)/2)+inputs

    class GELU(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function(0.5∗x∗(1+Tanh( (2/π)∗(x+0.044715∗x^3))) implemented by torch).
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = 0.5*inputs*(1+np.tanh((2/np.pi)*(inputs+0.044715*np.power(inputs,3))))

    class SELU(Activation):
        def __init__(self,alpha=1.6732,scale=1.0507):
            super().__init__()
            self.alpha = alpha
            self.scale = scale
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = self.scale*(np.max(0,inputs) + np.min(0,self.alpha*(np.exp(inputs)-1)))

    class BinaryStep(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.where(inputs<0,0,1)
        
    class HardSigmoid(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.maximum(0,np.minimum(1,0.2*inputs+0.5))

    class SoftSign(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = inputs/(1+np.abs(inputs))

    class ArcTan(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.arctan(inputs)

    class ISRLU(Activation):
        def __init__(self,alpha=1.0):
            super().__init__()
            self.alpha = alpha
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = inputs / np.sqrt(1+self.alpha*np.square(inputs))

    class SCALU(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.where(inputs<0,self.alpha*(np.exp(inputs)-1),inputs)

    class ReLU6(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.minimum(np.maximum(0,inputs),6)

    class ISRU(Activation):
        def __init__(self,alpha=1.0):
            super().__init__()
            self.alpha = alpha
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = inputs / np.sqrt(1+self.alpha*np.square(inputs))
    class SoftExponential(Activation):
        def __init__(self,alpha=1.0,beta=1.0):
            super().__init__()
            self.alpha = alpha
            self.beta = beta
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.where(inputs<0,-np.log(1-self.alpha*(inputs+self.beta))/self.alpha,inputs)
    
    class Sine(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.sin(inputs)

    class Triangular(Activation):
        def __init__(self,width=1.0):
            super().__init__()
            self.width = width
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.maximum(0,np.minimum(1,(1/self.width)*(inputs+self.width)))

    class BentIdentity2(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = ((np.sqrt(np.square(inputs)+1)-1)/2)+inputs

    class SQRT(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.sqrt(inputs)

    class ReSQRT(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.sqrt(np.maximum(0,inputs))-np.sqrt(np.maximum(0,-inputs))

    class LogSigmoid(Activation):
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.log(1/(1+np.exp(-inputs)))

    class RBF(Activation):
        def __init__(self,centers,gamma=1.0):
            super().__init__()
            self.centers = centers
            self.gamma = gamma
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.exp(-self.gamma*np.square(inputs-self.centers))

    class SqRBF(Activation):

        def __init__(self,gamma=1.0):
            super().__init__()
            self.gamma = gamma
        def forward(self,inputs):
            """
            This method is used to calculate the output of the activation function.
            Args:
                inputs: This is the input to the activation function.
            """
            self.output = np.exp(-self.gamma*np.square(inputs))

class LossFunctions():
    class Loss:

        def regularization_loss(self):
            regularization_loss = 0
            for layer in self.trainable_layers:
                if layer.weight_regularizer_l1 > 0:
                    regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
                
                if layer.weight_regularizer_l2 > 0:
                    regularization_loss += layer.weight_regularizer_l2 * np.sum(np.abs(layer.weights))

                if layer.bias_regularizer_l1 > 0:
                    regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
                if layer.bias_regularizer_l2 > 0:
                    regularization_loss += layer.bias_regularizer_l2 * np.sum(np.abs(layer.biases))

            return regularization_loss

        def remember_trainable_layers(self,trainable_layers):
            self.trainable_layers = trainable_layers

        def calculate(self,output,y,*,include_regularization=False):
            sample_losses = self.forward(output,y)
            data_loss = np.mean(sample_losses)
            self.accumulated_sum += np.sum(sample_losses)
            self.accumulated_count +=len(sample_losses)

            if not include_regularization:
                return data_loss
            return data_loss,self.regularization_loss()
        
        def calculate_accumulate(self,*,include_regularization=False):
            data_loss = self.accumulated_sum / self.accumulated_count

            if not include_regularization:
                return data_loss
            
            return data_loss, self.regularization_loss()
        
        def new_pass(self):
            self.accumulated_sum = 0
            self.accumulated_count = 0



    class MeanSquaredError(Loss):
        def forward(self,y_pred,y_true):
            sample_losses = np.mean((y_true-y_pred)**2,axis=-1)
            return sample_losses
        
        def backward(self,dvalues,y_true):
            samples=len(dvalues)
            outputs = len(dvalues[0])
            self.dinputs = -2*(y_true-dvalues)/outputs
            self.dinputs = self.dinputs/samples
    
    class MeanAbsoluteError(Loss):
        def forward(self,y_pred,y_true):
            sample_losses = np.mean(np.abs(y_true-y_pred),axis=-1)
            return sample_losses
        
        def backward(self,dvalues,y_true):
            samples=len(dvalues)
            outputs = len(dvalues[0])
            self.dinputs = np.sign(y_true-dvalues)/outputs
            self.dinputs = self.dinputs/samples

    class BinaryCrossEntropy(Loss):
        def forward(self,y_pred,y_true):
            y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
            sample_losses = -(y_true*np.log(y_pred_clipped)+(1-y_true)*np.log(1-y_pred_clipped))
            sample_losses = np.mean(sample_losses,axis=-1)
            return sample_losses
        
        def backward(self,dvalues,y_true):
            samples=len(dvalues)
            outputs = len(dvalues[0])
            clipped_dvalues = np.clip(dvalues,1e-7,1-1e-7)

            self.dinputs = -((y_true/clipped_dvalues) - (1-y_true) /(1-clipped_dvalues)) / outputs
            self.dinputs = self.dinputs/samples

    class CategoricalCrossEntropy(Loss):
        def forward(self,y_pred,y_true):
            samples = len(y_pred)
            y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
            if len(y_true.shape)==1:
                correct_confidences = y_pred_clipped[range(samples),y_true]
            elif len(y_true.shape)==2:
                correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
            negative_log_likelihoods = -np.log(correct_confidences)
            return negative_log_likelihoods
        
        def backward(self,dvalues,y_true):
            samples = len(dvalues)

            labels = len(dvalues[0])

            if len(y_true.shape) == 1:
                y_true = np.eye(labels,y_true)
            
            self.dinputs = (-y_true/ dvalues)/samples
    
    class SparseCategoricalCrossEntropy(Loss):
        def forward(self,y_pred,y_true):
            samples = len(y_pred)
            y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
            if len(y_true.shape)==1:
                correct_confidences = y_pred_clipped[range(samples),y_true]
            elif len(y_true.shape)==2:
                correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
            negative_log_likelihoods = -np.log(correct_confidences)
            return negative_log_likelihoods
    
    class KullbackLeiblerDivergence(Loss):
        def forward(self,y_pred,y_true):
            y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
            sample_losses = y_true*np.log(y_true/y_pred_clipped)
            return sample_losses
        
    class Hinge(Loss):
        def forward(self,y_pred,y_true):
            sample_losses = np.maximum(0,1-y_true*y_pred)
            return sample_losses
    

    class Huber(Loss):
        def __init__(self,delta=1.0):
            super().__init__()
            self.delta = delta
        def forward(self,y_pred,y_true):
            error = y_true-y_pred
            is_small_error = np.abs(error)<=self.delta
            squared_loss = np.square(error)/2
            linear_loss = self.delta*(np.abs(error)-self.delta/2)
            sample_losses = np.where(is_small_error,squared_loss,linear_loss)
            return sample_losses
    
    class LogCosh(Loss):
        def forward(self,y_pred,y_true):
            error = y_true-y_pred
            sample_losses = np.log(np.cosh(error))
            return sample_losses
    
    class CosineProximity(Loss):
        def forward(self,y_pred,y_true):
            y_pred_normalized = np.linalg.norm(y_pred, axis=-1, keepdims=True)
            y_pred_normalized = np.divide(y_pred, y_pred_normalized)
            y_true_normalized = np.linalg.norm(y_true, axis=-1, keepdims=True)
            y_true_normalized = np.divide(y_true, y_true_normalized)
            sample_losses = -np.sum(y_true_normalized*y_pred_normalized,axis=-1)
            return sample_losses


    class Triplet(Loss):
        def __init__(self,alpha=0.2):
            super().__init__()
            self.alpha = alpha
        def forward(self,y_pred,y_true):
            anchor,positive,negative = y_pred
            positive_distance = tf.reduce_mean(tf.square(anchor-positive),axis=-1)
            negative_distance = tf.reduce_mean(tf.square(anchor-negative),axis=-1)
            triplet_loss = tf.maximum(positive_distance-negative_distance+self.alpha,0.0)
            return triplet_loss
    class Center(Loss):
        def __init__(self,alfa=0.5):
            super().__init__()
            self.alfa = alfa
        def forward(self,y_pred,y_true):
            centers,y = y_pred
            sample_losses = tf.reduce_mean(tf.square(y-centers[y_true]),axis=-1)
            return sample_losses
    class Contastive(Loss):
        def __init__(self,margin=1.0):
            super().__init__()
            self.margin = margin
        def forward(self,y_pred,y_true):
            y_pred_left,y_pred_right = y_pred
            squared_pred = tf.square(y_pred_left-y_pred_right)
            squared_margin = tf.square(tf.maximum(self.margin-y_pred_left+y_pred_right,0))
            sample_losses = tf.reduce_mean(y_true*squared_pred+(1-y_true)*squared_margin,axis=-1)
            return sample_losses


    class Focal(Loss):
        def __init__(self,gamma=2.0,alpha=0.25):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
        def forward(self,y_pred,y_true):
            y_pred_clipped = tf.clip_by_value(y_pred,1e-7,1-1e-7)
            focal_loss = -tf.reduce_sum(self.alpha*tf.pow(1-y_pred_clipped,self.gamma)*tf.math.log(y_pred_clipped),axis=-1)
            return focal_loss
    
    class Dice(Loss):
        def __init__(self,smooth=1.0):
            super().__init__()
            self.smooth = smooth
        def forward(self,y_pred,y_true):
            y_pred_clipped = tf.clip_by_value(y_pred,1e-7,1-1e-7)
            numerator = 2*tf.reduce_sum(y_true*y_pred_clipped,axis=-1)
            denominator = tf.reduce_sum(y_true+y_pred_clipped,axis=-1)
            dice_loss = 1-numerator/(denominator+self.smooth)
            return dice_loss
    
    class Wasserstein(Loss):
        def forward(self,y_pred,y_true):
            return tf.reduce_mean(y_true*y_pred)
    
    class Adverserial(Loss):
        def __init__(self,base_loss,beta):
            super().__init__()
            self.base_loss = base_loss
            self.beta = beta
        def forward(self,y_pred,y_true):
            return self.base_loss(y_pred,y_true)-self.beta*self.base_loss(1-y_pred,y_true)
        
    class MutualInformation(Loss):
        def __init__(self):
            super().__init__()
        def forward(self,y_pred,y_true):
            return tf.reduce_mean(y_true*y_pred-tf.math.log(1+tf.exp(y_pred)))
    
    class SmoothL1(Loss):
        def forward(self,y_pred,y_true):
            error = tf.abs(y_true-y_pred)
            quadratic = tf.clip_by_value(error,0.0,1.0)
            linear = error-0.5
            sample_losses = tf.reduce_mean(tf.where(error<1.0,0.5*quadratic,linear),axis=-1)
            return sample_losses
    
    class SigmoidFocal(Loss):
        def __init__(self,gamma=2.0,alpha=0.25):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
        def forward(self,y_pred,y_true):
            y_pred_clipped = tf.clip_by_value(y_pred,1e-7,1-1e-7)
            focal_loss = -tf.reduce_sum(self.alpha*tf.pow(1-y_pred_clipped,self.gamma)*tf.math.log(y_pred_clipped),axis=-1)
            return focal_loss
    class LovaszHinge(Loss):
        def forward(self,y_pred,y_true):
            y_pred = tf.clip_by_value(y_pred,-1.0,1.0)
            errors = 1-y_true*y_pred
            errors_sorted = tf.sort(errors,axis=-1,direction='DESCENDING')
            sample_losses = tf.reduce_mean(errors_sorted[:,0],axis=-1)
            return sample_losses
    
    class Tversky(Loss):
        def __init__(self,alpha=0.5,beta=0.5,smooth=1.0):
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.smooth = smooth
        def forward(self,y_pred,y_true):
            y_pred_clipped = tf.clip_by_value(y_pred,1e-7,1-1e-7)
            numerator = tf.reduce_sum(y_true*y_pred_clipped,axis=-1)
            denominator = numerator+self.alpha*tf.reduce_sum(y_true*(1-y_pred_clipped),axis=-1)+self.beta*tf.reduce_sum((1-y_true)*y_pred_clipped,axis=-1)
            tversky_loss = 1-numerator/(denominator+self.smooth)
            return tversky_loss
    
    class Jaccard(Loss):
        def __init__(self,smooth=1.0):
            super().__init__()
            self.smooth = smooth
        def forward(self,y_pred,y_true):
            y_pred_clipped = tf.clip_by_value(y_pred,1e-7,1-1e-7)
            intersection = tf.reduce_sum(y_true*y_pred_clipped,axis=-1)
            union = tf.reduce_sum(y_true+y_pred_clipped,axis=-1)-intersection
            jaccard_loss = 1-(intersection+self.smooth)/(union+self.smooth)
            return jaccard_loss
    
    class WeightedBCE(Loss):
        def __init__(self,weights):
            super().__init__()
            self.weights = weights
        def forward(self,y_pred,y_true):
            y_pred_clipped = tf.clip_by_value(y_pred,1e-7,1-1e-7)
            sample_losses = tf.reduce_mean(self.weights*(y_true*tf.math.log(y_pred_clipped)+(1-y_true)*tf.math.log(1-y_pred_clipped)),axis=-1)
            return sample_losses
    
    class ContrastiveWithMargin(Loss):
        def __init__(self,margin):
            super().__init__()
            self.margin = margin
        def forward(self,y_pred,y_true):
            square_pred = tf.square(y_pred)
            margin_square = tf.square(tf.maximum(self.margin-y_pred,0))
            sample_losses = tf.reduce_mean(y_true*square_pred+(1-y_true)*margin_square,axis=-1)
            return sample_losses
    
    class HuberSiamese(Loss):
        def __init__(self,delta):
            super().__init__()
            self.delta = delta
        def forward(self,y_pred,y_true):
            error = tf.abs(y_true-y_pred)
            quadratic = tf.clip_by_value(error,0.0,self.delta)
            linear = error-self.delta
            sample_losses = tf.reduce_mean(0.5*tf.square(quadratic)+self.delta*linear,axis=-1)
            return sample_losses
    
    class BinaryFocal(Loss):
        def __init__(self,gamma=2.0,alpha=0.25):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
        def forward(self,y_pred,y_true):
            y_pred_clipped = tf.clip_by_value(y_pred,1e-7,1-1e-7)
            focal_loss = -tf.reduce_sum(self.alpha*tf.pow(1-y_pred_clipped,self.gamma)*tf.math.log(y_pred_clipped)+(1-self.alpha)*tf.pow(y_pred_clipped,self.gamma)*tf.math.log(1-y_pred_clipped),axis=-1)
            return focal_loss
    
    class MultiLabelSoftMargin(Loss):
        def forward(self,y_pred,y_true):
            y_pred_clipped = tf.clip_by_value(y_pred,1e-7,1-1e-7)
            sample_losses = tf.reduce_mean(tf.math.log(1+tf.exp(-y_true*y_pred_clipped)),axis=-1)
            return sample_losses
    class Logit(Loss):
        def forward(self,y_pred,y_true):
            y_pred_clipped = tf.clip_by_value(y_pred,1e-7,1-1e-7)
            sample_losses = tf.reduce_mean(tf.math.log(1+tf.exp(-y_true*y_pred_clipped)),axis=-1)
            return sample_losses
        
class Activation_S_Loss_CC():
    def __init__(self):
        self.activation = ActivationFunctions.Softmax()
        self.loss = LossFunctions.CategoricalCrossEntropy()
    
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output,y_true)
    def backward(self,dvalues,y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizers():
    class Optimizer():
        def __init__(self,learning_rate= 1.0,decay=0.0,):
            self.learning_rate = learning_rate
            self.current_learning_rate = learning_rate
            self.decay = decay
            self.iterations = 0

        def pre_update_params(self):
            if self.decay:
                self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

        def post_update_params(self):
            self.iterations += 1
    
    class SGD(Optimizer):
        def __init__(self, learning_rate=1, decay=0.,momentum=0.):
            super().__init__(learning_rate, decay)
            self.momentum = momentum
        
        def update_params(self, layer):
            if self.momentum:
                if not hasattr(layer,'weight_momentums'):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.biases)
                
                weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
                layer.weight_momentums = weight_updates

                bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
                layer.bias_momentums = bias_updates
            else:
                weight_updates = -self.current_learning_rate * layer.dweights
                bias_updates = -self.current_learning_rate * layer.dbiases
            
            layer.weights += weight_updates
            layer.biases += bias_updates

    class AdaGrad(Optimizer):
        def __init__(self, learning_rate=1, decay=0,epsilon = 1e-7):
            super().__init__(learning_rate, decay)
            self.epsilon = epsilon

        # Update parameters
        def update_params(self, layer):

            # If layer does not contain cache arrays,
            # create them filled with zeros
            if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            # Update cache with squared current gradients
            layer.weight_cache += layer.dweights**2
            layer.bias_cache += layer.dbiases**2

            # Vanilla SGD parameter update + normalization
            # with square rooted cache
            layer.weights += -self.current_learning_rate *layer.dweights /(np.sqrt(layer.weight_cache) + self.epsilon)
            layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    class RMSProp(Optimizer):
        def __init__(self,learning_rate=1.0,decay=0.0,epsilon=1e-7,rho=0.9):
            super().__init__(learning_rate,decay)
            self.epsilon = epsilon
            self.rho = rho
        def update_params(self,layer):
            if not hasattr(layer,'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)
            
            layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho)*layer.dweights**2
            layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho)*layer.dbiases**2

            layer.weights += -self.current_learning_rate *layer.dweights/(np.sqrt(layer.weight_cache) + self.epsilon)
            layer.biases += -self.current_learning_rate *layer.dbiases /(np.sqrt(layer.bias_cache) + self.epsilon)

    class Adam(Optimizer):
        def __init__(self,learning_rate=0.001, decay = 0.,epsilon=1e-7,beta_1= 0.9,beta_2=0.999):
            super().__init__(learning_rate,decay)
            self.epsilon = epsilon
            self.beta_1 = beta_1
            self.beta_2 = beta_2
        def update_params(self,layer):

            if not hasattr(layer,'weight_cache'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                layer.bias_cache = np.zeros_like(layer.biases)
            
            layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1)*layer.dweights
            layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1)*layer.dbiases

            weight_momentums_corrected = layer.weight_momentums/(1-self.beta_1**(self.iterations+1))
            bias_momentums_corrected = layer.bias_momentums/(1-self.beta_1**(self.iterations+1))
            
            layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2)*layer.dweights**2
            layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2)*layer.dbiases**2

            weight_cache_corrected = layer.weight_cache/(1-self.beta_2**(self.iterations+1))
            bias_cache_corrected = layer.bias_cache/(1-self.beta_2**(self.iterations+1))

            layer.weights += -self.current_learning_rate * weight_momentums_corrected/(np.sqrt(weight_cache_corrected) + self.epsilon)
            layer.biases += -self.current_learning_rate * bias_momentums_corrected/(np.sqrt(bias_cache_corrected) + self.epsilon)

class Model():
    def __init__(self):
        self.layers = []
    
    def add(self,layer):
        self.layers.append(layer)

    def set(self,*,loss,optimizer,accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    def train(self,X,y,*,epochs=1,print_every=1,validation_data=None):


        self. accuracy.init(y)
        for epoch in range(1,epochs+1):
            output = self.forward(X,training=True)

            data_loss, regularization_loss = self.loss.calculate(output,y,include_regularization=True)
            loss = data_loss + regularization_loss
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions,y)
            self.backward(output,y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                        f'acc: {accuracy:.3f}, ' +
                        f'loss: {loss:.3f} (' +
                        f'data_loss: {data_loss:.3f}, ' +
                        f'reg_loss: {regularization_loss:.3f}), ' +
                        f'lr: {self.optimizer.current_learning_rate}')
                
        
        if validation_data is not None:
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            X_val,y_val = validation_data
            output = self.forward(X_val,training=False)
            loss = self.loss.calculate(output,y_val)
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions,y_val)
            print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

    
    def finalize(self):
        self.input_layer = Layer.Input()

        layer_count = len(self.layers)

        self.trainable_layers = []
        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            
            if hasattr(self.layers[i],'weights'):
                self.trainable_layers.append(self.layers[i])
        
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1],ActivationFunctions.Softmax) and isinstance(self.loss, LossFunctions.CategoricalCrossentropy):
            self.softmax_classifier_output = LossFunctions.Softmax_CategoricalCrossentropy()


    def forward(self,X,training):
        self.input_layer.forward(X,training)
        for layer in self.layers:
            layer.forward(layer.prev.output,training)
        return layer.output
    
    def backward(self,output,y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output,y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output,y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self,X_val,y_val,*,batch_size =None):
        validation_steps = 1
        if batch_size is not None:

            validation_steps=len(X_val) //batch_size
            if validation_steps*batch_size<len(X_val):
                validation_steps +=1

class Accuracy():

    class Main():
        def calculate(self,predictions,y):
            comparisons = self.compare(predictions,y)
            accuracy = np.mean(comparisons)
            self.accumulated_sum += np.sum(comparisons)
            self.accumulated_count += len(comparisons)
            return accuracy
    
        def calculate_accumulated(self):
            accuracy = self.accumulated_sum/self.accumulated_count
            return accuracy
        
        def new_pass(self):
            self.accumulated_sum = 0
            self.accumulated_count = 0

    class Regression(Main):
        def __init__(self):
            self.precision = None
        
        def init(self,y,reinint = False):
            if self.precision is None or reinint:
                self.precision = np.std(y)/250
        
        def compare(self,predictions,y):
            return np.absolute(predictions - y) < self.precision
    
    class Categorical(Main):
        def init(self,y):
            pass
        
        def compare(self,predictions,y):
            if len(y.shape) == 2:
                y = np.argmax(y,axis=1)
            
            return predictions == y

#install opencv with 

    
## Data Loading

import os 
import cv2

def load_mnist_dataset(dataset,path):

    labels = os.listdir(os.path.join(path,dataset))
    X = []
    y = []

    for label in labels:
        for file in os.listdir(os.path.join(path,dataset,label)):
            image = cv2.imread(os.path.join(path,dataset,label,file),cv2.IMREAD_UNCHANGED)

            X.append(image)
            y.append(label)
    
    return np.array(X),np.array(y).astype('uint8')

def create_data_mnist(path):
    X,y = load_mnist_dataset('train',path)
    X_test,y_test = load_mnist_dataset('test',path)
    return X,y,X_test,y_test

X,y,X_test,y_test = create_data_mnist('fashion_mnist_images')

X = (X.astype(np.float32) - 127.5)/127.5
X_test = (X_test.astype(np.float32) - 127.5)/127.5

print(X.min(),X.max())

print(X.shape)

example = np.array([[1,2],[3,4]])
flattened = example.reshape(-1)

print(example)
print(example.shape)

print(flattened)
print(flattened.shape)

"""
print(labels)

import cv2
image_data = cv2.imread('fashion_mnist_images/train/7/0002.png', cv2.IMREAD_UNCHANGED)

print(image_data)

np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(image_data) 
plt.show()
"""

"""
X,y = sine_data()
dense1 = Layer.Dense(1, 64)

model = Model()
model.add(Layer.Dense(1,64))
model.add(ActivationFunctions().ReLU())
model.add(Layer.Dense(64,64))
model.add(ActivationFunctions().ReLU())
model.add(Layer.Dense(64,1))
model.add(ActivationFunctions().Linear())

model.set(loss= LossFunctions.MeanSquaredError(),optimizer=Optimizers.Adam(learning_rate=0.005,decay=1e-3),accuracy=Accuracy.Regression())
print(model.layers)

model.finalize()
model.train(X,y,epochs=10000,print_every=100)
"""
#dropout1= Layer.Dropout(0.1)
"""
dense2 = Layer.Dense(64,64)

dense3 = Layer.Dense(64,1)
"""

"""
relu1 = ActivationFunctions().ReLU()
relu2 = ActivationFunctions().ReLU()
linear = ActivationFunctions.Linear()

MSE= LossFunctions.MeanSquaredError()
optimizer = Optimizers.Adam(learning_rate=0.005,decay=1e-3)

accuracy_precision = np.std(y)/250

for epoch in range(10001):

    dense1.forward(X)
    relu1.forward(dense1.output)
    dense2.forward(relu1.output)
    relu2.forward(dense2.output)
    dense3.forward(relu2.output)
    linear.forward(dense3.output)
    data_loss = MSE.calculate(linear.output,y)

    regularization_loss = MSE.regularization_loss(dense1) + MSE.regularization_loss(dense2) + MSE.regularization_loss(dense3)
    loss = data_loss + regularization_loss

    preds = linear.output.copy()
    accuracy = np.mean(np.abs(preds - y) < accuracy_precision)

    if not epoch % 100:
        print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate:.3f}")



    MSE.backward(linear.output,y)

    linear.backward(MSE.dinputs)
    dense3.backward(linear.dinputs)
    relu2.backward(dense3.dinputs)
    dense2.backward(relu2.dinputs)
    relu1.backward(dense2.dinputs)
    
    dense1.backward(relu1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)

    optimizer.post_update_params()
 
X_test,y_test = sine_data()
y_test = y_test.reshape(-1,1)
dense1.forward(X_test)
relu1.forward(dense1.output)
dense2.forward(relu1.output)
relu2.forward(dense2.output)
dense3.forward(relu2.output)
linear.forward(dense3.output)


import matplotlib.pyplot as plt
plt.plot(X_test,y_test)
plt.plot(X_test,linear.output)
plt.show()"""