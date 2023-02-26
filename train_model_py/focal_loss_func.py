import tensorflow as tf
import keras.backend as K


def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        """
        Focal loss function for multi-class classification problems.
        gamma: A positive focusing parameter. A lower value of gamma will focus more on the easy examples.
        alpha: A balancing parameter between the positive and negative class examples. A value of 0.5 gives equal weightage.
        """
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate cross-entropy loss
        ce_loss = -y_true * K.log(y_pred)

        # Calculate the weight matrix for the focal loss
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        weight_matrix = alpha * y_true * K.pow(1 - pt, gamma)

        # Apply the weight matrix to the cross-entropy loss
        focal_loss = ce_loss * weight_matrix

        # Calculate the mean loss over all examples
        loss = K.mean(K.sum(focal_loss, axis=-1))

        return loss

    return focal_loss_fixed
