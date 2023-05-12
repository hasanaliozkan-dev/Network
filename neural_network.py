import nnfs 
from nnfs.datasets import spiral_data
nnfs.init()
import numpy as np
class Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        
    
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

    

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
            pass

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
            self.output = np.maximum(0,inputs)

    class Softmax(Activation):
        """
        This class implements the Softmax activation function.
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
            exp_values = np.exp(inputs)
            self.output = exp_values/np.sum(exp_values,axis=1,keepdims=True)

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
            self.output = 1/(1+np.exp(-inputs))

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
        




X,y = spiral_data(samples=100 , classes = 3)
dense1 = Dense(2,3)
dense2 = Dense(3,3)
relu = ActivationFunctions().ReLU()
softmax = ActivationFunctions.Softmax()
cce_loss = LossFunctions().CategoricalCrossEntropy()
dense1.forward(X)
relu.forward(dense1.output)
dense2.forward(relu.output)
softmax.forward(dense2.output)
loss = cce_loss.calculate(softmax.output,y)

preds = np.argmax(softmax.output,axis=1)
if len(y.shape)==2:
    y = np.argmax(y,axis=1)
accuracy = np.mean(preds==y)
print(accuracy)
print(loss)


