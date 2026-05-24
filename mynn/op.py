from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # self.W = initialize_method(size=(in_dim, out_dim))
        # self.b = initialize_method(size=(1, out_dim))
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        output = X @ self.W + self.b
        return output
        pass

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = grad.shape[0]

        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)

        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        output = grad @ self.W.T
        return output
        pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None: 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.W = np.random.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size
        ) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        self.b = np.zeros((out_channels,))

        self.grads = {
            'W': None,
            'b': None
        }

        self.params = {
            'W': self.W,
            'b': self.b
        }

        self.input = None
        self.input_padded = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        #pass
    

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X

        batch_size, in_channels, H, W = X.shape
        k = self.kernel_size
        s = self.stride
        p = self.padding

        if p > 0:
            X_padded = np.pad(
                X,
                ((0, 0), (0, 0), (p, p), (p, p)),
                mode='constant'
            )
        else:
            X_padded = X

        self.input_padded = X_padded

        _, _, H_padded, W_padded = X_padded.shape

        out_H = (H_padded - k) // s + 1
        out_W = (W_padded - k) // s + 1

        output = np.zeros((batch_size, self.out_channels, out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                h_start = i * s
                h_end = h_start + k
                w_start = j * s
                w_end = w_start + k

                patch = X_padded[:, :, h_start:h_end, w_start:w_end]

                output[:, :, i, j] = np.tensordot(
                    patch,
                    self.W,
                    axes=([1, 2, 3], [1, 2, 3])
                ) + self.b

        return output
        #pass

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X_padded = self.input_padded
        batch_size, in_channels, H_padded, W_padded = X_padded.shape

        k = self.kernel_size
        s = self.stride
        p = self.padding

        _, out_channels, out_H, out_W = grads.shape

        dX_padded = np.zeros_like(X_padded)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        db = np.sum(grads, axis=(0, 2, 3))

        for i in range(out_H):
            for j in range(out_W):
                h_start = i * s
                h_end = h_start + k
                w_start = j * s
                w_end = w_start + k

                patch = X_padded[:, :, h_start:h_end, w_start:w_end]
                grad_ij = grads[:, :, i, j]

                dW += np.tensordot(
                    grad_ij,
                    patch,
                    axes=([0], [0])
                )

                dX_padded[:, :, h_start:h_end, w_start:w_end] += np.tensordot(
                    grad_ij,
                    self.W,
                    axes=([1], [0])
                )

        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        self.grads['W'] = dW
        self.grads['b'] = db

        if p > 0:
            dX = dX_padded[:, :, p:-p, p:-p]
        else:
            dX = dX_padded

        return dX
        pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None
        self.optimizable = False
        #pass

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.predicts = predicts
        self.labels = labels.astype(int)

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        batch_size = predicts.shape[0]
        eps = 1e-12

        correct_probs = self.probs[np.arange(batch_size), self.labels]
        loss = -np.mean(np.log(correct_probs + eps))

        return loss
        pass
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        batch_size = self.predicts.shape[0]

        if self.has_softmax:
            self.grads = self.probs.copy()
            self.grads[np.arange(batch_size), self.labels] -= 1
            self.grads /= batch_size
        else:
            eps = 1e-12
            self.grads = np.zeros_like(self.predicts)
            self.grads[np.arange(batch_size), self.labels] = -1 / (self.probs[np.arange(batch_size), self.labels] + eps)
            self.grads /= batch_size

        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition