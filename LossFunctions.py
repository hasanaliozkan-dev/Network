import numpy as np
import tensorflow as tf

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
        