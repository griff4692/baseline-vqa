from keras import backend as K
import numpy as np

margin = 0.9

EMBEDDING_MAT = np.load('./data/top_answer_embeddings') # embed_mat, embed_size
SIZE_EMBED_MAT = len(EMBEDDING_MAT)

embedding_mat = K.variable(EMBEDDING_MAT) # embed_mat, embed_size

def my_hinge(y_true, y_pred):
    y_true_norm = K.l2_normalize(y_true, axis=-1) # batch_size, embed_size
    y_pred_norm = K.l2_normalize(y_pred, axis=-1)

    # compute similarity
    sim_true_pred = K.batch_dot(y_true_norm, y_pred_norm, axes=-1) # batch_size, 1 
    sim_true_pred_repeated = K.repeat(sim_true_pred, SIZE_EMBED_MAT) # batch_size, embed_mat, 1

    y_pred_norm_repeated = K.repeat(y_pred_norm, SIZE_EMBED_MAT) # batch_size, embed_mat, embed_size

    sim_false_pred_repeated = y_pred_norm_repeated * embedding_mat # batch_size, embed_mat, embed_size

    sim_false_pred_sum = K.sum(sim_false_pred_repeated, keepdims=True, axis=-1)

    delta = margin + sim_false_pred_sum - sim_true_pred_repeated

    max_margin = K.maximum(delta, 0.0)

    max_margin_sum = K.sum(max_margin, axis=-1)

    return K.mean(max_margin_sum, axis=-1)
