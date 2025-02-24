import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization

@tf.keras.utils.register_keras_serializable()
def focal_loss(gamma=2., alpha=0.25):
    """
    Compute focal loss for multi-class classification.

    Parameters:
    gamma (float): Focusing parameter.
    alpha (float): Balancing parameter.

    Returns:
    function: Loss function.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
        
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        
        return K.mean(K.sum(fl, axis=-1))
    return focal_loss_fixed

@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(Layer):
    """
    Multi-Head Self Attention Layer.
    """
    def __init__(self, embed_dim=256, num_heads=8, dropout_rate=0.1, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension {embed_dim} must be divisible by number of heads {num_heads}")

        self.projection_dim = embed_dim // num_heads
        
        # Dense layers for query, key, and value
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        
        # Dense layer to combine heads
        self.combine_heads = Dense(embed_dim)

        # Dropout and Layer Normalization
        self.dropout = Dropout(dropout_rate)
        self.layernorm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Compute scaled dot-product attention
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        attention = tf.matmul(weights, value)

        attention = self.combine_heads(attention)
        output = self.dropout(attention)
        return self.layernorm(inputs + output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        })
        return config


# Load the trained Keras model
model_path = "DenseNet201.keras"  # Update if needed

try:
    model = load_model(model_path, custom_objects={"MultiHeadSelfAttention": MultiHeadSelfAttention, "focal_loss_fixed": focal_loss()})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

model.summary()

# Define class labels (update as necessary)
class_labels = ["No_Fall", "Fall"]

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Resize frame to match DenseNet201 input size (224x224)
    resized_frame = cv2.resize(frame, (128, 128))

    # Convert BGR to RGB (since OpenCV reads in BGR format)
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Normalize & preprocess input
    processed_frame = preprocess_input(rgb_frame)  # Normalizes input as per DenseNet201
    processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(processed_frame)
    predicted_class = np.argmax(predictions)  # Get class index
    confidence = np.max(predictions) * 100  # Get confidence percentage
    predicted_label = class_labels[predicted_class]  # Get class label

    # Display prediction on frame
    text = f'Prediction: {predicted_label} ({confidence:.2f}%)'
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Webcam Feed - Fall Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
