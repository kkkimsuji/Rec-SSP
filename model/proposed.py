import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, GRU, Concatenate, Multiply, Add, 
    Dropout, LayerNormalization, MultiHeadAttention, Layer, Lambda
)

def ModelBuild(n, multi_hot_len, config):
    """
    Builds the Rec-SSP Model Architecture.
    Hyperparameters are pulled from the config dictionary.
    """
    # Extract hyperparameters from config
    num_heads = config.get('num_heads', 8)
    key_dim = config.get('key_dim', 128)
    dff = config.get('dff', 2048)
    dropout_rate = config.get('dropout_rate', 0.1)
    epsilon = config.get('epsilon', 1e-6)

    # --- 1. Input Layers ---
    user_static_input = Input(shape=(768, ), dtype='float32', name='user_static_input')
    item_bert_input = Input(shape=(768, ), dtype='float32', name='item_bert_input')
    item_label_input = Input(shape=(multi_hot_len, ), dtype='float32', name='item_label_input')
    user_dynamic_input = Input(shape=(n, 768, ), dtype='float32', name='user_dynamic_input')
    item_label_seq_input = Input(shape=(n, multi_hot_len, ), dtype='float32', name='item_label_seq_input')

    # --- 2. Item Feature Extraction ---
    item_label_feature = Dense(units=key_dim, activation='relu')(item_label_input)
    item_bert_feature = Dense(units=key_dim, activation='relu')(item_bert_input)
    item_feature = Concatenate()([item_label_feature, item_bert_feature])
    item_feature = Dense(units=key_dim, activation='relu', name='item_MLP')(item_feature)

    # --- 3. User Long-term Preference (Static) ---
    user_static_feature = Dense(units=key_dim, activation='relu', name='user_MLP')(user_static_input)

    # --- 4. User Short-term Preference (Dynamic) ---
    # Sequential processing using GRU
    user_dynamic_review = GRU(key_dim, return_sequences=False)(user_dynamic_input)
    user_dynamic_label = GRU(key_dim, return_sequences=False)(item_label_seq_input)
    
    user_dynamic_feature = Concatenate()([user_dynamic_review, user_dynamic_label])
    user_dynamic_feature = Dense(units=key_dim, activation='relu', name='user_dynamic_MLP')(user_dynamic_feature)

    # --- 5. Gated Fusion Mechanism ---
    user_static_tanh = Dense(units=key_dim, activation='tanh')(user_static_feature)
    user_dynamic_tanh = Dense(units=key_dim, activation='tanh')(user_dynamic_feature)

    # Calculate Sigmoid Gate to balance static and dynamic signals
    gate_input = Concatenate()([user_static_tanh, user_dynamic_tanh])
    gate_user_feature = Dense(units=key_dim, activation='sigmoid', name='Gate')(gate_input)

    # Apply gating logic
    gated_static = Multiply()([gate_user_feature, user_static_tanh])
    gated_dynamic = Multiply()([1 - gate_user_feature, user_dynamic_tanh])
    user_feature_fusion = Add(name='Fusion')([gated_static, gated_dynamic])

    # --- 6. Final Rating Prediction ---
    overall = Concatenate()([user_feature_fusion, item_feature])
    mlp_overall = Dense(units=key_dim, activation='relu')(overall)
    mlp_overall = Dropout(dropout_rate)(mlp_overall)
    mlp_overall = Dense(units=64, activation='relu')(mlp_overall)
    output = Dense(units=1, name='Output')(mlp_overall)

    return Model(inputs=[user_static_input, item_bert_input, item_label_input, 
                        user_dynamic_input, item_label_seq_input], outputs=output)

class SelfAttentionBlock(Layer):
    """
    Custom Layer for Self-Attention based sequence modeling.
    """
    def __init__(self, config, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.num_heads = config.get('num_heads', 8)
        self.key_dim = config.get('key_dim', 128)
        self.dff = config.get('dff', 2048)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        self.epsilon = config.get('epsilon', 1e-6)

        self.multi_head_attention = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim)
        self.dropout = Dropout(self.dropout_rate)
        self.add = Add()
        self.layer_norm = LayerNormalization(epsilon=self.epsilon)
        self.ffn = tf.keras.Sequential([
            Dense(self.dff, activation='relu'),
            Dense(self.key_dim)
        ])
        self.reshape_layer = Lambda(lambda x: tf.reshape(x, [-1, tf.shape(x)[1], self.key_dim]))

    def call(self, inputs):
        inputs_reshaped = self.reshape_layer(inputs)
        # Attention + Residual Connection
        attn_output = self.multi_head_attention(inputs_reshaped, inputs_reshaped, inputs_reshaped)
        attn_output = self.dropout(attn_output)
        res1 = self.layer_norm(self.add([attn_output, inputs_reshaped]))

        # FFN + Residual Connection
        ffn_output = self.ffn(res1)
        ffn_output = self.dropout(ffn_output)
        return self.layer_norm(self.add([ffn_output, res1]))