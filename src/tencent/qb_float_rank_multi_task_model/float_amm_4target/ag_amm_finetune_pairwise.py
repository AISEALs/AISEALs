#!/usr/bin/python3

import tensorflow as tf
import tensorflow.keras as K

import numerous
import model_zoo.layers
import mlp
from model_zoo.inputs import Inputs
from model_zoo.models import NumerousModel
from layers_new import FM, MultiHeadSelfAttention
from autogroup import AutoGroupLayer

from numerous.train.run_config import RunConfig
from numerous.utils.config import get_config

debug_mode = int(get_config().get_option("userdefine.config.debug_mode", default_value="0"))
debug_mode = bool(debug_mode)
amm_weight = float(get_config().get_option("userdefine.config.amm_weight", default_value="1.0"))
denoise_weight = float(get_config().get_option("userdefine.config.denoise_weight", default_value="1.0"))
pairwise_weight = float(get_config().get_option("userdefine.config.pairwise_weight", default_value="1.0"))
print('amm_weight:', amm_weight)
print('denoise_weight: {}'.format(denoise_weight))
print('pairwise_weight: {}'.format(pairwise_weight))

from config.gpu_config import build_gpu_conf
GPU_CONFIG = build_gpu_conf()

#################配置信息，超参数#########################
# embedding all
all_user_slots = {201, 202, 203, 204, 205, 206, 209, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224,
                  225, 226, 227, 228, 232, 233, 234, 235,
                  236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 252, 253, 254, 255, 266,
                  267, 268, 269, 270, 271, 272, 273, 274,
                  275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 289, 290, 291, 292, 293, 294, 295, 296,
                  297, 298, 1711}
all_extra_user_slots = {1403, 1404, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614}
user_base_feature_names = ['sparse_w_' + str(slot_id) for slot_id in all_user_slots]
user_base_extra_feature_names = ['sparse_w_' + str(slot_id) for slot_id in all_extra_user_slots]

all_doc_slots = {401, 402, 403, 404, 405, 406, 407, 408, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423,
                 424, 425, 426, 427, 428, 429, 430, 432,
                 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452,
                 453, 454, 455, 456, 457, 458, 459, 460}
extra_doc_slots = {480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490,
                   491, 492, 493, 494, 495}
doc_base_feature_names = ['sparse_w_' + str(slot_id) for slot_id in all_doc_slots]
doc_extra_base_feature_names = ['sparse_w_' + str(slot_id) for slot_id in extra_doc_slots]
# fm embedding sum
doc_fm_slots = {401, 402, 403, 404, 405, 406, 407, 408, 412, 413, 414, 415, 416, 417, 418, 419, 420, 435, 457, 458, 459,
                460, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489,
                490, 491, 492, 493, 494, 495}
doc_fm_feature_names = ['sparse_fm_w_' + str(slot_id) for slot_id in doc_fm_slots]

user_fm_slots = {202, 203, 204, 205, 206, 209, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225,
                 226, 227, 228,
                 232, 233, 234, 235, 236, 237, 238, 239, 240, 255, 266, 267, 289, 290, 291, 292, 293, 294, 295, 296,
                 297, 298, 1403, 1404, 1711, 2606, 2607,
                 2608, 2609, 2610, 2611, 2612, 2613, 2614}
user_fm_feature_names = ['sparse_fm_w_' + str(slot_id) for slot_id in user_fm_slots]
# seq
seq_item_slots = {
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79}

seq_item_feature_names = ['sparse_w_' + str(slot_id) for slot_id in seq_item_slots]

seq_pos_slots = {
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
    123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
    146, 147, 148, 149}
seq_pos_feature_names = ['sparse_w_' + str(slot_id) for slot_id in seq_pos_slots]

new_add_user_slots = [2784, 2785, 2786, 2788, 2789]
new_add_user_feature_names = ['new_add_user_sparse_w_' + str(slot_id) for slot_id in new_add_user_slots]

inherit_slots_mapping = {218: 2781, 219: 2782, 220: 2783, 286: 2787, 427: 3011, 438: 3012, 453: 3013, 354: 3014,
                         455: 3015, 456: 3016}

# seq attention
num_heads = 2
num_blocks = 1
attention_embedding_size = 16
att_user_context_slots = {201, 202, 203, 221}
user_att_feature_names = ['sparse_att_w_' + str(slot_id) for slot_id in att_user_context_slots]

att_doc_slots = {401, 407, 412, 457, 458, 459}
doc_att_feature_names = ['sparse_att_w_' + str(slot_id) for slot_id in att_doc_slots]

autogroup_slots = [201, 202, 203, 204, 209, 211, 213, 216, 219, 220, 221, 222, 223, 226, 227, 228,
                   232, 255, 266, 267, 284, 285, 286, 289, 290, 291, 292, 293, 294, 295, 296, 297,
                   298, 1403, 1404, 1711, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614]

autogroup_feature_names = ['sparse_w_' + str(slot_id) for slot_id in autogroup_slots]
ORDERS = [2, 3, 4]
GROUP_NUMS = [20, 10, 5]

# dense embedding
EMBEDDING_DIM = 8

label_configs = [['play_time', 10000, 1, 1],
                 ['video_time', 10011, 0, 0],
                 ['noise_flag', 10000, 8, 8],
                 ['fresh_num', 10012, 0, 0],
                 ['query_mask_slot', 30000, 0, 0],
                 ['sample_valid_slot', 30001, 0, 0],
                 ]

task_num = 4
task_names = ['skip', 'finish', 'ratio', 'time']
expert_num = 5
bins = [15, 30, 46, 93, 125]
bins_fresh = [1.5, 2.5, 3.5, 4.5, 10.5]

use_bn = True


def compute_covariance(X):
    # Center the data
    X_centered = X - tf.reduce_mean(X, axis=0)  # [num_samples_i, num_features]
    # Calculate covariance matrix
    cov = tf.matmul(tf.transpose(X_centered), X_centered) / tf.cast(tf.shape(X_centered)[0] - 1,
                                                                    tf.float32)  # [num_features, num_features]
    return cov


def compute_loss(X, y, major_category):
    # X: [num_samples, num_features]
    # y: [num_samples]
    min_label = tf.reduce_min(y)  # find the minimal category label
    y = y - min_label  # adjust the category labels to start from 0
    major_category = major_category - min_label  # adjust the major category accordingly

    # Number of categories
    num_categories = tf.reduce_max(y) + 1

    # Partition X into sub-matrices according to the category labels in y
    partitions = tf.dynamic_partition(X, y, num_categories)  # list of tensors with shape [num_samples_i, num_features]

    # Calculate covariance matrices for each category
    covariances = [compute_covariance(partitions[i]) for i in range(num_categories)]

    # Calculate loss for each non-major category
    losses = [tf.norm(covariances[major_category] - covariances[i], ord='fro') ** 2 for i in range(num_categories) if
              i != major_category]

    # Sum up the losses
    total_loss = tf.add_n(losses)  # total_loss: scalar

    return total_loss


def compute_comparisons_matrix(input_tensor, apply_sign=False):
    """
    计算输入张量的每一对元素之间的比较，返回一个上三角矩阵，
    如果元素i大于元素j，输出1，如果相等，输出0，如果小于，输出-1
    """
    # 将输入张量的形状从 [batch_size, 1] 转为 [batch_size]
    input_tensor = tf.squeeze(input_tensor, axis=-1)

    # 计算元素之间的比较
    comparisons_matrix = tf.subtract(tf.expand_dims(input_tensor, axis=0), tf.expand_dims(input_tensor, axis=1))
    if apply_sign:
        comparisons_matrix = tf.sign(comparisons_matrix)

    # 获取上三角部分，其余部分设为 0
    # upper_triangle_comparisons = tf.linalg.band_part(comparisons_matrix, 0, -1)
    #
    # return upper_triangle_comparisons
    return comparisons_matrix


def generate_mask_matrix(input_tensor):
    """
    生成一个 mask 矩阵，如果输入张量的两个元素相等，则对应位置为1，否则为0。
    返回一个上三角矩阵。
    """
    # 将输入张量的形状从 [batch_size, 1] 转为 [batch_size]
    input_tensor = tf.squeeze(input_tensor, axis=-1)

    # 计算元素之间是否相等
    equal_matrix = tf.math.equal(tf.expand_dims(input_tensor, axis=0), tf.expand_dims(input_tensor, axis=1))

    # 获取上三角部分，其余部分设为 False
    # upper_triangle_mask = tf.linalg.band_part(equal_matrix, 0, -1)

    # return tf.cast(upper_triangle_mask, tf.float32)
    return equal_matrix


def cal_pairwise_loss(label, logits, mask_matrix):
    """
    计算 pairwise loss。
    只计算 comparisons_matrix 中对应 mask_matrix 为 1 的位置的元素的 loss。
    """
    label_cmp_matrix = compute_comparisons_matrix(label, apply_sign=True)
    pred_cmp_matrix = compute_comparisons_matrix(tf.sigmoid(logits), apply_sign=False)
    pair_wise_matrix_loss = tf.math.log_sigmoid(label_cmp_matrix * pred_cmp_matrix * tf.cast(mask_matrix, tf.float32))

    loss = tf.math.reduce_mean(pair_wise_matrix_loss, axis=1, keepdims=True)
    return loss

class DenoiseLayer(K.layers.Layer):
    def __init__(self, **kwargs):
        super(DenoiseLayer, self).__init__(**kwargs)
        self.mlp = model_zoo.layers.MLP(hidden_units=[512, 256, 32, 1], use_bn=use_bn)
        self.model = None

    def call(self, inputs, training):
        user_embedding, doc_embedding = inputs
        feature_input = tf.stop_gradient(tf.concat([user_embedding, doc_embedding], axis=1))
        denoise_logits = self.mlp(feature_input, training=training)
        self.model = denoise_logits
        return self.model


class QBMiniFLoatModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.doc_experts = []
        for i in range(expert_num):
            expert = mlp.MLP(hidden_units=[512, 256, 128], use_bn=use_bn, name=f"base_doc_expert_{i}")
            self.doc_experts.append(expert)
        self.doc_gates = []
        self.doc_extra_gates = []
        for j in range(task_num):
            gate = K.layers.Dense(units=expert_num, activation=None, name=f"base_doc_gate_{j}")
            extra_gate = K.layers.Dense(units=expert_num, activation=None, name=f"base_doc_extra_gate_{j}")
            self.doc_gates.append(gate)
            self.doc_extra_gates.append(extra_gate)

        self.user_experts = []
        for i in range(expert_num):
            expert = mlp.MLP(hidden_units=[512, 256, 128], use_bn=use_bn, name=f"base_user_expert_{i}")
            self.user_experts.append(expert)
        self.user_gates = []
        self.user_extra_gates = []
        for j in range(task_num):
            gate = K.layers.Dense(units=expert_num, activation=None, name=f"base_user_gate_{j}")
            extra_gate = K.layers.Dense(units=expert_num, activation=None, name=f"base_user_extra_gate_{j}")
            self.user_gates.append(gate)
            self.user_extra_gates.append(extra_gate)

        self.doc_fm = FM(name="doc_fm")
        self.user_fm = FM(name="user_fm")

        self.fm_doc_mlp = model_zoo.layers.MLP(hidden_units=[32, 32], use_bn=use_bn, name="fm_doc")
        self.fm_user_mlp = model_zoo.layers.MLP(hidden_units=[32, 32], use_bn=use_bn, name="fm_user")

        self.user_att_layer = MultiHeadSelfAttention(attention_embedding_size=attention_embedding_size,
                                                     num_heads=num_heads,
                                                     num_blocks=num_blocks)

        self.user_attention_mlp = model_zoo.layers.MLP(hidden_units=[32], name="user_attention")
        self.doc_attention_mlp = model_zoo.layers.MLP(hidden_units=[32, 32], use_bn=use_bn, name="doc_attention")

        self.doc_task_towers = []
        for j in range(task_num):
            tower = mlp.MLP(hidden_units=[128, 64, 32], use_bn=use_bn, name=f"doc_tower_{j}")
            self.doc_task_towers.append(tower)

        self.user_task_towers = []
        for j in range(task_num):
            tower = mlp.MLP(hidden_units=[128, 64, 32], use_bn=use_bn, name=f"user_tower_{j}")
            self.user_task_towers.append(tower)

        self.denoise_net_finish = DenoiseLayer(name="denoise_net_finish")
        self.denoise_net_skip = DenoiseLayer(name="denoise_net_skip")

        self.auto_group_layers = []
        for order, group_num in zip(ORDERS, GROUP_NUMS):
            self.auto_group_layers.append(
                AutoGroupLayer(slot_num=len(autogroup_feature_names), group_num=group_num, order=order))

    def call(self, inputs, training):
        # doc mmoe
        doc_merge_layers = tf.concat([inputs[feature_name] for feature_name in doc_base_feature_names], axis=1)
        doc_extra_merge_layers = tf.concat([inputs[feature_name] for feature_name in doc_extra_base_feature_names],
                                           axis=1)
        expert_net_doc = []
        for doc_expert in self.doc_experts:
            base_doc_layer = doc_expert(doc_merge_layers, doc_extra_merge_layers, training)
            expert_net_doc.append(base_doc_layer)
        expert_net_doc = tf.stack(expert_net_doc)
        expert_net_doc = tf.transpose(expert_net_doc, perm=[0, 2, 1])
        dnn_gate_doc = []
        for doc_gate, doc_extra_gate in zip(self.doc_gates, self.doc_extra_gates):
            expert_weight = tf.nn.softmax(doc_gate(doc_merge_layers) + doc_extra_gate(doc_extra_merge_layers), -1)
            gate_net = tf.einsum('ai,ija->aj', expert_weight, expert_net_doc)
            dnn_gate_doc.append(gate_net)

        # user mmoe
        expert_net_user = []
        user_merge_layers = tf.concat([inputs[feature_name] for feature_name in user_base_feature_names], axis=1)
        autogroup_features = tf.stack([inputs[feature_name] for feature_name in autogroup_feature_names], axis=1)
        new_add_user_features = tf.concat([inputs[feature_name] for feature_name in new_add_user_feature_names], axis=1)
        print("autogroup_features shape:", autogroup_features.shape)  # (None, 9, 8)
        auto_group_outputs = []
        for i in range(len(self.auto_group_layers)):
            # [batch_size, group_num, embedding_dim]
            auto_group_output = self.auto_group_layers[i](autogroup_features, training)
            print("auto_group_output shape:", auto_group_output.shape)  # (None, 9, 8)
            _, w, h = auto_group_output.shape
            auto_group_output_layer = tf.reshape(auto_group_output, [-1, w * h])
            auto_group_outputs.append(auto_group_output_layer)
        auto_group_layer = tf.concat(auto_group_outputs, axis=1)
        print(f"auto_group_layer shape: {auto_group_layer.shape}")

        user_extra_merge_layers = tf.concat(
            [inputs[feature_name] for feature_name in user_base_extra_feature_names] + [auto_group_layer],
            axis=1)

        for user_expert in self.user_experts:
            base_user_layer = user_expert(user_merge_layers, [user_extra_merge_layers, new_add_user_features], training)
            expert_net_user.append(base_user_layer)
        expert_net_user = tf.stack(expert_net_user)
        expert_net_user = tf.transpose(expert_net_user, perm=[0, 2, 1])
        dnn_gate_user = []
        for user_gate, user_extra_gate in zip(self.user_gates, self.user_extra_gates):
            expert_weight = tf.nn.softmax(user_gate(user_merge_layers) + user_extra_gate(user_extra_merge_layers), -1)
            gate_net = tf.einsum('ai,ija->aj', expert_weight, expert_net_user)
            dnn_gate_user.append(gate_net)

        doc_fm_inputs = [inputs[feature_name] for feature_name in doc_fm_feature_names]
        doc_fm_embedding_sum_layer = self.doc_fm(doc_fm_inputs)

        user_fm_inputs = [inputs[feature_name] for feature_name in user_fm_feature_names]
        user_fm_embedding_sum_layer = self.user_fm(user_fm_inputs)

        fm_doc_layer = self.fm_doc_mlp(doc_fm_embedding_sum_layer, training)
        fm_user_layer = self.fm_user_mlp(user_fm_embedding_sum_layer, training)

        seq_item_inputs = tf.stack([inputs[feature_name] for feature_name in seq_item_feature_names], axis=1)
        seq_pos_inputs = tf.stack([inputs[feature_name] for feature_name in seq_pos_feature_names], axis=1)
        user_attention_layer = tf.concat([seq_item_inputs, seq_pos_inputs], axis=-1)
        attention_output = self.user_att_layer(user_attention_layer, training)

        user_att_inputs = tf.stack([inputs[feature_name] for feature_name in user_att_feature_names], axis=1)
        att_user_context_layer = tf.reduce_sum(user_att_inputs, axis=1)
        translation_output = tf.concat([att_user_context_layer, attention_output], axis=1)
        user_attention = self.user_attention_mlp(translation_output, training)

        doc_att_inputs = tf.stack([inputs[feature_name] for feature_name in doc_att_feature_names], axis=1)
        att_doc_fm_embedding_sum_layer = tf.reduce_sum(doc_att_inputs, axis=1)
        att_doc_layer = self.doc_attention_mlp(att_doc_fm_embedding_sum_layer, training)

        dnn_multitask_doc = []
        for j, task in enumerate(task_names):
            dnn_net = tf.concat([dnn_gate_doc[j], att_doc_layer, fm_doc_layer], axis=1)
            dnn_out = self.doc_task_towers[j](dnn_net, inputs[f'i_amm_{task}'], training)
            dnn_multitask_doc.append(dnn_out)

        dnn_multitask_user = []
        for j, task in enumerate(task_names):
            dnn_net = tf.concat([dnn_gate_user[j], user_attention, fm_user_layer], axis=1)
            dnn_out = self.user_task_towers[j](dnn_net, inputs[f'u_amm_{task}'], training)
            dnn_multitask_user.append(dnn_out)

        user_vec = tf.concat([dnn_multitask_user[2],
                              dnn_multitask_user[0],
                              dnn_multitask_user[1],
                              dnn_multitask_user[3], ], axis=1)
        doc_vec = tf.concat([dnn_multitask_doc[2],
                             dnn_multitask_doc[0],
                             dnn_multitask_doc[1],
                             dnn_multitask_doc[3], ], axis=1)

        user_vec = tf.identity(user_vec, name="user_vec_infer")
        doc_vec = tf.identity(doc_vec, name="doc_vec_infer")

        inner_product = tf.multiply(dnn_multitask_user[0], dnn_multitask_doc[0])
        skip_logits = tf.reduce_sum(inner_product, 1, keepdims=True)
        skip_pred = tf.sigmoid(skip_logits, name='skip_pred')

        inner_product = tf.multiply(dnn_multitask_user[1], dnn_multitask_doc[1])
        finish_logits = tf.reduce_sum(inner_product, 1, keepdims=True)
        finish_pred = tf.sigmoid(finish_logits, name='finish_pred')

        inner_product = tf.multiply(dnn_multitask_user[2], dnn_multitask_doc[2])
        ratio_logits = tf.reduce_sum(inner_product, 1, keepdims=True)
        ratio_pred = tf.sigmoid(ratio_logits, name='ratio_pred')

        inner_product = tf.multiply(dnn_multitask_user[3], dnn_multitask_doc[3])
        time_logits = tf.reduce_sum(inner_product, 1, keepdims=True)
        time_pred = tf.sigmoid(time_logits, name='time_pred')

        denoise_finish_logits = self.denoise_net_finish([user_merge_layers, doc_merge_layers], training)
        denoise_skip_logits = self.denoise_net_skip([user_merge_layers, doc_merge_layers], training)

        model_outputs = {'user_vec_infer': user_vec,
                         'doc_vec_infer': doc_vec,
                         'denoise_finish_logits': denoise_finish_logits,
                         'denoise_skip_logits': denoise_skip_logits,
                         "skip_logits": skip_logits,
                         'skip_pred': skip_pred,
                         'finish_logits': finish_logits,
                         'finish_pred': finish_pred,
                         "time_logits": time_logits,
                         'time_pred': time_pred,
                         'ratio_logits': ratio_logits,
                         'ratio_pred': ratio_pred,
                         }
        for i, task in enumerate(task_names):
            model_outputs.update({f'u_amm_{task}': inputs[f'u_amm_{task}']})
            model_outputs.update({f'i_amm_{task}': inputs[f'i_amm_{task}']})
            model_outputs.update({f'user_vec_{task}': dnn_multitask_user[i]})
            model_outputs.update({f'doc_vec_{task}': dnn_multitask_doc[i]})

        print('model_outputs:', model_outputs)

        return model_outputs


class Runner(NumerousModel):
    def __init__(self, name):
        super().__init__(name)
        self.add_metric('auc_skip', numerous.metrics.AUC(name='auc_skip'))
        self.add_metric('auc_finish', numerous.metrics.AUC(name='auc_finish'))
        self.add_metric('ratio_loss', numerous.metrics.Mean(name='ratio_loss_mse'))
        self.add_metric('time_loss', numerous.metrics.Mean(name='time_loss_mse'))
        self.add_metric('final_loss', numerous.metrics.Mean(name='final_loss'))
        self.add_metric('denoise_loss_skip', numerous.metrics.Mean(name='denoise_loss_skip'))
        self.add_metric('denoise_loss_finish', numerous.metrics.Mean(name='denoise_loss_finish'))

        self.add_metric('pointwise_loss', numerous.metrics.Mean(name='pointwise_loss'))
        self.add_metric('pairwise_loss', numerous.metrics.Mean(name='pairwise_loss'))
        self.add_metric('amm_loss', numerous.metrics.Mean(name='amm_loss'))
        self.add_metric('denoise_loss', numerous.metrics.Mean(name='denoise_loss'))

        for i, task in enumerate(task_names):
            self.add_metric(f'amm_loss_{task}', numerous.metrics.Mean(name=f'amm_loss_{task}'))

        optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=0.0002)
        self.set_optimizer(optimizer)
        self.dnn = QBMiniFLoatModel(name="qbminifloat_model")

    def build_inputs(self):
        inputs = Inputs(optimizer=numerous.optimizers.Adam(learning_rate=0.001))
        emb_initializer = numerous.initializers.RandomUniform(minval=-0.001, maxval=0.001)
        emb_optimizer = numerous.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        combiner = numerous.layers.Combiner.SUM
        for slot_id, feature_name in zip(all_doc_slots, doc_base_feature_names):
            if slot_id in inherit_slots_mapping:
                slot_id = inherit_slots_mapping[slot_id]
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=4 * EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )
        for slot_id, feature_name in zip(extra_doc_slots, doc_extra_base_feature_names):
            if slot_id in inherit_slots_mapping:
                slot_id = inherit_slots_mapping[slot_id]
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=4 * EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )

        for slot_id, feature_name in zip(doc_fm_slots, doc_fm_feature_names):
            if slot_id in inherit_slots_mapping:
                slot_id = inherit_slots_mapping[slot_id]
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=4 * EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )

        for slot_id, feature_name in zip(att_doc_slots, doc_att_feature_names):
            if slot_id in inherit_slots_mapping:
                slot_id = inherit_slots_mapping[slot_id]
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=4 * EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )

        for slot_id, feature_name in zip(all_user_slots, user_base_feature_names):
            if slot_id in inherit_slots_mapping:
                slot_id = inherit_slots_mapping[slot_id]
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=2 * EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )
        for slot_id, feature_name in zip(all_extra_user_slots, user_base_extra_feature_names):
            if slot_id in inherit_slots_mapping:
                slot_id = inherit_slots_mapping[slot_id]
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=2 * EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )

        for slot_id, feature_name in zip(user_fm_slots, user_fm_feature_names):
            if slot_id in inherit_slots_mapping:
                slot_id = inherit_slots_mapping[slot_id]
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=2 * EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )

        for slot_id, feature_name in zip(att_user_context_slots, user_att_feature_names):
            if slot_id in inherit_slots_mapping:
                slot_id = inherit_slots_mapping[slot_id]
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=2 * EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )

        for slot_id, feature_name in zip(seq_item_slots, seq_item_feature_names):
            if slot_id in inherit_slots_mapping:
                slot_id = inherit_slots_mapping[slot_id]
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )

        for slot_id, feature_name in zip(seq_pos_slots, seq_pos_feature_names):
            if slot_id in inherit_slots_mapping:
                slot_id = inherit_slots_mapping[slot_id]
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )

        for slot_id, feature_name in zip(new_add_user_slots, new_add_user_feature_names):
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )

        # amm feature
        for i, task in enumerate(task_names):
            inputs.add_sparse_feature(
                feature_name=f"u_amm_{task}",
                slot_id=201,  # uid
                embedding_dim=32,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )
            inputs.add_sparse_feature(
                feature_name=f"i_amm_{task}",
                slot_id=401,  # doc_id
                embedding_dim=32,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )

        for name, slot_id, lower_bound, upper_bound in label_configs:
            inputs.add_label(label_name=name, slot_id=slot_id,
                             lower_bound=lower_bound, upper_bound=upper_bound)

        embedding, configs = inputs.build()
        return embedding, configs

    def call(self, inputs, training):
        inputs = self.preprocess_inputs(inputs)  # process the features into a dictionary
        outputs = self.dnn(inputs, training)
        return outputs

    def compute_loss(self, labels, model_outputs, training):
        playtime = tf.clip_by_value(labels["play_time"], clip_value_min=0.0, clip_value_max=3600)
        video_time = tf.clip_by_value(labels["video_time"], clip_value_min=1.0, clip_value_max=3600)
        noise_flag = labels['noise_flag']
        y_data = tf.clip_by_value(labels['label'], 0, 1)

        is_no_noise = tf.greater(noise_flag, 0.5 * tf.ones_like(noise_flag))

        sample_weights = tf.ones_like(playtime, tf.float32)
        sample_weights = tf.where(tf.less(video_time, 5 * tf.ones_like(playtime, tf.float32)),
                                  tf.zeros_like(playtime, tf.float32), sample_weights)  # 物理时长小于5s丢掉
        sample_weights = tf.where(tf.less(playtime, tf.zeros_like(playtime, tf.float32)),
                                  tf.zeros_like(playtime, tf.float32), sample_weights)  # 丢掉播放时长小于0
        sample_weights = tf.where(tf.greater(video_time, 300 * tf.ones_like(playtime, tf.float32)),
                                  tf.zeros_like(playtime, tf.float32), sample_weights)  # 大于250s也不要

        is_finish = tf.where(tf.greater_equal(playtime, video_time), tf.ones_like(playtime), tf.zeros_like(playtime))
        is_skip = tf.where(tf.less(playtime, 4.5 * tf.ones_like(playtime)), tf.ones_like(playtime),
                           tf.zeros_like(playtime))
        time_label = tf.clip_by_value(playtime / 110., 0, 1)

        # ratio label
        ratio_label = playtime / (video_time + 1e-8)
        ratio_label = tf.clip_by_value(ratio_label, 0.0, 1.0, name="ratio_label_clip")
        # ratio_label = tf.cast(ratio_label * 3.0, tf.float32, name="ratio_label")

        denoise_finish_logits = model_outputs['denoise_finish_logits']
        denoise_weight_finish = tf.stop_gradient(tf.math.sigmoid(denoise_finish_logits))
        denoise_skip_logits = model_outputs['denoise_skip_logits']
        denoise_weight_skip = 1 - tf.stop_gradient(tf.math.sigmoid(denoise_skip_logits))

        sample_weights_finish = tf.where(is_no_noise, sample_weights, denoise_weight_finish * sample_weights)
        sample_weights_skip = tf.where(is_no_noise, sample_weights, denoise_weight_skip * sample_weights)

        skip_logits = model_outputs['skip_logits']
        finish_logits = model_outputs['finish_logits']
        ratio_logits = model_outputs['ratio_logits']
        time_logits = model_outputs['time_logits']

        skip_pred = tf.sigmoid(skip_logits)
        finish_pred = tf.sigmoid(finish_logits)
        ratio_pred = tf.sigmoid(ratio_logits)
        time_pred = tf.sigmoid(time_logits)

        finish_loss_ce = tf.compat.v1.losses.sigmoid_cross_entropy(is_finish, finish_logits, sample_weights_finish,
                                                                   reduction=tf.compat.v1.losses.Reduction.NONE)
        skip_loss_ce = tf.compat.v1.losses.sigmoid_cross_entropy(is_skip, skip_logits, sample_weights_skip,
                                                                 reduction=tf.compat.v1.losses.Reduction.NONE)
        ratio_loss_ce = tf.compat.v1.losses.sigmoid_cross_entropy(ratio_label, ratio_logits, sample_weights_finish,
                                                                  reduction=tf.compat.v1.losses.Reduction.NONE)
        time_loss_ce = tf.compat.v1.losses.sigmoid_cross_entropy(time_label, time_logits, sample_weights_finish,
                                                                 reduction=tf.compat.v1.losses.Reduction.NONE)
        ratio_loss_mse = tf.compat.v1.losses.mean_squared_error(ratio_label, ratio_pred, sample_weights_finish,
                                                                reduction=tf.compat.v1.losses.Reduction.NONE)
        time_loss_mse = tf.compat.v1.losses.mean_squared_error(time_label, time_pred, sample_weights_finish,
                                                               reduction=tf.compat.v1.losses.Reduction.NONE)
        denoise_loss_skip = tf.compat.v1.losses.sigmoid_cross_entropy(is_skip, denoise_skip_logits,
                                                                      tf.where(is_no_noise, tf.ones_like(y_data) / (
                                                                                  tf.reduce_mean(tf.cast(is_no_noise,
                                                                                                         tf.float32)) + 1e-9),
                                                                               tf.zeros_like(y_data)),
                                                                      reduction=tf.compat.v1.losses.Reduction.NONE)
        denoise_loss_finish = tf.compat.v1.losses.sigmoid_cross_entropy(is_finish, denoise_finish_logits,
                                                                        tf.where(is_no_noise, tf.ones_like(y_data) / (
                                                                                    tf.reduce_mean(tf.cast(is_no_noise,
                                                                                                           tf.float32)) + 1e-9),
                                                                                 tf.zeros_like(y_data)),
                                                                        reduction=tf.compat.v1.losses.Reduction.NONE)
        amm_losses = []
        for i, task in enumerate(task_names):
            user_vec = model_outputs[f'user_vec_{task}']
            doc_vec = model_outputs[f'doc_vec_{task}']
            u_amm = model_outputs[f'u_amm_{task}']
            i_amm = model_outputs[f'i_amm_{task}']
            user_amm_loss = tf.reduce_sum(tf.square(tf.stop_gradient(user_vec) - i_amm), axis=-1, keepdims=True)
            doc_amm_loss = tf.reduce_sum(tf.square(tf.stop_gradient(doc_vec) - u_amm), axis=-1, keepdims=True)
            amm_losses.append((user_amm_loss + doc_amm_loss) * 0.5)

        amm_loss = tf.math.add_n(amm_losses)
        is_positive_sample = tf.where(tf.greater(playtime, video_time), 1.0, 0.0)  # 完成率超过100%的部分当成正样本
        amm_loss = tf.multiply(is_positive_sample, amm_loss)

        dummy_sample_mask = tf.cast(labels['sample_valid_slot'], tf.bool)
        query_mask_matrix = generate_mask_matrix(labels['query_mask_slot'])

        is_skip_bpr_loss = cal_pairwise_loss(is_skip, skip_logits, query_mask_matrix)
        finish_bpr_loss = cal_pairwise_loss(is_finish, finish_logits, query_mask_matrix)
        ratio_bpr_loss = cal_pairwise_loss(ratio_label, ratio_logits, query_mask_matrix)
        time_bpr_loss = cal_pairwise_loss(time_label, time_logits, query_mask_matrix)

        # 1. pintwise loss
        pointwise_loss = skip_loss_ce + finish_loss_ce + ratio_loss_ce + time_loss_ce
        pointwise_loss = tf.boolean_mask(pointwise_loss, dummy_sample_mask)
        # 2. denoise loss
        denoise_loss = denoise_loss_finish + denoise_loss_skip
        denoise_loss = tf.boolean_mask(denoise_loss, dummy_sample_mask) * denoise_weight
        # 3. pairwise loss
        pairwise_loss = is_skip_bpr_loss + finish_bpr_loss + ratio_bpr_loss + time_bpr_loss
        pairwise_loss = tf.boolean_mask(pairwise_loss, dummy_sample_mask) * pairwise_weight
        # 4. amm loss
        amm_loss = tf.boolean_mask(amm_loss, dummy_sample_mask) * amm_weight

        final_loss = pointwise_loss + denoise_loss + pairwise_loss + amm_loss

        # tf.print('sample_valid_slot:', labels['sample_valid_slot'],
        #          'query_mask_slot:', labels['query_mask_slot'],
        #          'query_mask_matrix:', query_mask_matrix,
        #          'is_skip_bpr_loss:', is_skip_bpr_loss
        #          )

        self.get_metric("auc_skip").update_state(is_skip, skip_pred)
        self.get_metric("auc_finish").update_state(is_finish, finish_pred)
        self.get_metric('ratio_loss').update_state(ratio_loss_mse)
        self.get_metric('time_loss').update_state(time_loss_mse)
        self.get_metric("final_loss").update_state(final_loss)
        self.get_metric("denoise_loss_skip").update_state(denoise_loss_skip)
        self.get_metric("denoise_loss_finish").update_state(denoise_loss_finish)
        for i, task in enumerate(task_names):
            self.get_metric(f'amm_loss_{task}').update_state(amm_losses[i])

        self.get_metric("pointwise_loss").update_state(pointwise_loss)
        self.get_metric("pairwise_loss").update_state(pairwise_loss)
        self.get_metric("amm_loss").update_state(amm_loss)
        self.get_metric("denoise_loss").update_state(denoise_loss)

        return final_loss


numerous.cluster.start(numerous.cluster.get_cluster_config(run_mode=numerous.cluster.RunMode.CLUSTER))
with numerous.distribute.get_strategy().scope():
    model = Runner(name="gpu_runner")
    model.init_by_run_config(RunConfig(GPU_CONFIG))
    model.run()
numerous.cluster.stop()
