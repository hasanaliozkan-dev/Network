from Layer import Layer
import copy
import pickle
import numpy as np
from ActivationFunctions import ActivationFunctions
from LossFunctions import LossFunctions

class Model():
    def __init__(self):
        self.layers = []
    
    def add(self,layer):
        self.layers.append(layer)

    def set(self,*,loss = None,optimizer = None,accuracy = None):
        
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if accuracy is not None:
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

    def save_parameters(self,path):
        with open(path,'wb') as f:
            pickle.dump(self.get_parameters(),f)

    def load_parameters(self,path):
        with open(path,'rb') as f:
            self.set_parameters(pickle.load(f))

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
        
            if self.loss is not None:
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

    def get_parameters(self):
        parameters = []

        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters

    def set_parameters(self,parameters):

        for parameter_set,layer in zip(parameters,self.trainable.layers):
            layer.set_parameters(*parameter_set)

    def save(self,path):
        model = copy.deepcopy(self)

        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop('output',None)
        model.loss.__dict__.pop('dinputs',None)

        for layer in model.layers:
            for property in ['inputs','output','dinputs','dweights','dbiases']:
                layer.__dict__.pop(property,None)
        
        with open(path,'wb') as f:
            pickle.dump(model,f)
    
    def predict(self, X , * , batch_size= None):
        prediction_steps = 1

        if batch_size is not None:
            prediction_steps = len(X) //batch_size

            if prediction_steps*batch_size < len(X):
                prediction_steps +=1

        output = []

        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X 

            else:
                batch_X = X[step*batch_size:(step+1)*batch_size] 

                batch_output = self.forward(batch_X,training=False)

                output.append(batch_output)

        return np.vstack(output)
                   


    @staticmethod
    def load(path):
        with open(path,'rb') as f:
            model = pickle.load(f)
        return model
