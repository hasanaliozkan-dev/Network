import numpy as np 
from LossFunctions import LossFunctions

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
