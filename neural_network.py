import nnfs 
from nnfs.datasets import spiral_data
nnfs.init()
import numpy as np

class Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        
    
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases

    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

class ActivationFunctions:

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

class LossFunctions:
    class Loss:
        def calculate(self,output,y):
            sample_losses = self.forward(output,y)
            data_loss = np.mean(sample_losses)
            return data_loss

    class MeanSquaredError(Loss):
        def forward(self,y_pred,y_true):
            sample_losses = np.mean((y_true-y_pred)**2,axis=-1)
            return sample_losses
    
    class BinaryCrossEntropy(Loss):
        def forward(self,y_pred,y_true):
            y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
            sample_losses = -(y_true*np.log(y_pred_clipped)+(1-y_true)*np.log(1-y_pred_clipped))
            return sample_losses
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
    
    class MeanAbsoluteError(Loss):
        def forward(self,y_pred,y_true):
            sample_losses = np.mean(np.abs(y_true-y_pred),axis=-1)
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

X,y = spiral_data(samples=100 , classes = 3)
dense1 = Dense(2,64)
dense2 = Dense(64,3)
"""
print("Dense1 weights: ",dense1.weights)
print()
print("Dense1 biases: ",dense1.biases)
print()
print("Dense2 weights: ",dense2.weights)
print()
print("Dense2 biases: ",dense2.biases)
print()
"""
relu = ActivationFunctions().ReLU()
softcc = Activation_S_Loss_CC()
optimizer = Optimizers.Adam(learning_rate=0.05,decay=5e-7)


for epoch in range(10001):

    dense1.forward(X)
    relu.forward(dense1.output)
    dense2.forward(relu.output)
    loss = softcc.forward(dense2.output,y)
    preds = np.argmax(softcc.output,axis=1)
    if len(y.shape)==2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(preds==y)

    if not epoch % 100:
        print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.current_learning_rate:.3f}")


    softcc.backward(softcc.output,y)
    dense2.backward(softcc.dinputs)
    relu.backward(dense2.dinputs)
    dense1.backward(relu.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

    optimizer.post_update_params()
 
X_test,y_test = spiral_data(samples=100,classes=3)
dense1.forward(X_test)
relu.forward(dense1.output)
dense2.forward(relu.output)
loss = softcc.forward(dense2.output,y_test)
preds = np.argmax(softcc.output,axis=1)
if len(y_test.shape)==2:
    y_test = np.argmax(y_test,axis=1)
accuracy = np.mean(preds==y_test)
print(f"validation, acc: {accuracy:.3f}, loss: {loss:.3f}")
