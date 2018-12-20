import tensorflow as tf


# Key=>Tensorboard上での表示名
# Value=>Tensor
_summary_variable_dict = {}


def add_to_watch_list(summary_name, tensor):
    _summary_variable_dict[summary_name] = tensor

def classification_loss(logits, labels):
    """classificationで生じる損失

    :param tf.Tensor logits: NC tf.float32
    :param tf.Tensor labels: NC tf.int32
    :return:
    """
    #class_loss_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="class_loss")
    #class_loss = tf.reduce_mean(class_loss_vector, name="class_loss")
    class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    return class_loss
