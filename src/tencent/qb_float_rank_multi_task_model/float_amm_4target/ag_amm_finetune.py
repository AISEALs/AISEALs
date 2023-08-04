#!/usr/bin/python3

import os
import subprocess
import sys

subprocess.call(["sudo", "/usr/bin/python3", "-m", "pip", "install", "numerous-modelzoo-gpu==0.5.8"])

os.system("sudo chown -R mqq:mqq /tmp/hadooplogutil.log")
os.environ["CLASSPATH"] = subprocess.check_output(
    ['/usr/local/hadoop-venus/bin/hadoop', 'classpath', '--glob']).decode('utf-8').strip()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import tensorflow.keras as K

tf.get_logger().setLevel("ERROR")

import numerous
import model_zoo.layers
import mlp
from model_zoo.inputs import Inputs
from model_zoo.models import KerasModel, NumerousModel
# from mlp import MLP
# from model_zoo.layers import MLP
from layers_new import FM, MultiHeadSelfAttention
from autogroup import AutoGroupLayer

#################GPU训练配置#########################
# GPU_CONFIG全部配置参见: https://iwiki.woa.com/pages/viewpage.action?pageId=1524230667
GPU_CARD_NUM = numerous.distribute.local_gpu_num()
GPU_CONFIG = {
    "is_auto_optimize": False,
    "reserved_gpu_memory_bytes": 1024 * 1024 * 1024 * 4,
    "max_gpu_memory_occupied_percentage": 0.8,
    "reader_config": {
        "one_ins_max_fea_num": 2048,
        "is_ignore_zero_key": False,
        "is_ignore_negative_factor_key": False,
        "is_ignore_non_batch_ins": False,
        "file_loader_thread_num": 32 * GPU_CARD_NUM,
        "extract_unique_key_thread_num": 32 * GPU_CARD_NUM,
        "merge_thread_num": 8 * GPU_CARD_NUM,
        "write_fileins_thread_num": 4 * GPU_CARD_NUM,
        "write_unique_key_thread_num": 4 * GPU_CARD_NUM,
        # "sub_dir_file_regexs": [".*part-000.*", ".*part-001.*", ".*part-002.*", ".*part-003.*", ".*part-004.*", ".*part-005.*", ".*part-006.*", ".*part-007.*", ".*part-008.*", ".*part-009.*"],
        "pass_dir_regexs": [".*"],
        "dir_format": "%Y-%m-%d/%H",
        "data_format": "text",
        "sample_col_delimiter": "|",
        "sample_feature_delimiter": ";",
        "feature_section_delimiter": ":",
        "id_col_pos": 0,
        "label_col_pos": 1,
        "feature_col_pos": 2,
        "plugin_path": "./libdefault_sample_preprocessor.so",
        # "plugin_path": "",
        "plugin_config": {},
    },
    "parameter_server_config": {
        "part_num": 1024,
        "hbm_max_key_num": 1024 * 1024 * 80,
        "hbm_max_dynamic_byte_size": 1024 * 1024 * 1024 * 20,
        "hbm_hashmap_load_factor": 0.75,
        "prepare_pass_thread_num": 12 * GPU_CARD_NUM,
        "build_pass_thread_num": 12 * GPU_CARD_NUM,
        "build_group_thread_num": 12 * GPU_CARD_NUM,
        "build_group_postprocess_thread_num": 4 * GPU_CARD_NUM,
        "do_persistence_thread_num": 24 * GPU_CARD_NUM,
        "load_checkpoint_thread_num": 24 * GPU_CARD_NUM,
        "use_parallel_optimizer": False,
        "delete_after_unseen_days": 7,
        "cow_clip": {"r": 1.0, "lower_bound": 1e-5}
    },
    "avengers_manager_base_config": {
        "pipeline_batch_num": 2,
        "read_ins_thread_num": 4 * GPU_CARD_NUM,
        "convert_ins_thread_num": 8 * GPU_CARD_NUM,
        "compute_batch_thread_num": 1,
    },
    "saver_config": {
        "model_dump_interval": 1,
        "recover_nn_by_name": True
    },
}

print("GPU_CONFIG:", GPU_CONFIG, flush=True)

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
print('user_base_feature_names:', user_base_feature_names)
print('user_base_extra_feature_names:', user_base_extra_feature_names)


all_doc_slots = {401, 402, 403, 404, 405, 406, 407, 408, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423,
                 424, 425, 426, 427, 428, 429, 430, 432,
                 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452,
                 453, 454, 455, 456, 457, 458, 459, 460}
extra_doc_slots = {480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490,
                   491, 492, 493, 494, 495}
doc_base_feature_names = ['sparse_w_' + str(slot_id) for slot_id in all_doc_slots]
doc_extra_base_feature_names = ['sparse_w_' + str(slot_id) for slot_id in extra_doc_slots]
print('doc_base_feature_names:', doc_base_feature_names)
print('doc_extra_base_feature_names:', doc_extra_base_feature_names)
for slot_id, feature_name in zip(all_doc_slots, doc_base_feature_names):
    print(slot_id, feature_name)

# fm embedding sum
doc_fm_slots = {401, 402, 403, 404, 405, 406, 407, 408, 412, 413, 414, 415, 416, 417, 418, 419, 420, 435, 457, 458, 459,
                460, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489,
                490, 491, 492, 493, 494, 495}
doc_fm_feature_names = ['sparse_fm_w_' + str(slot_id) for slot_id in doc_fm_slots]
print('doc_fm_feature_names:', doc_fm_feature_names)

user_fm_slots = {202, 203, 204, 205, 206, 209, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225,
                 226, 227, 228,
                 232, 233, 234, 235, 236, 237, 238, 239, 240, 255, 266, 267, 289, 290, 291, 292, 293, 294, 295, 296,
                 297, 298, 1403, 1404, 1711, 2606, 2607,
                 2608, 2609, 2610, 2611, 2612, 2613, 2614}
user_fm_feature_names = ['sparse_fm_w_' + str(slot_id) for slot_id in user_fm_slots]
print('user_fm_feature_names', user_fm_feature_names)
# seq
seq_item_slots = {
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79}

seq_item_feature_names = ['sparse_w_' + str(slot_id) for slot_id in seq_item_slots]
print('seq_item_feature_names:', seq_item_feature_names)

seq_pos_slots = {
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
    123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
    146, 147, 148, 149}
seq_pos_feature_names = ['sparse_w_' + str(slot_id) for slot_id in seq_pos_slots]
print('seq_pos_feature_names', seq_pos_feature_names)

new_add_user_slots = [2784,2785,2786,2788,2789]
new_add_user_feature_names = ['new_add_user_sparse_w_' + str(slot_id) for slot_id in new_add_user_slots]
print('new_add_user_feature_names:', new_add_user_feature_names)

inherit_slots_mapping = {218:2781, 219:2782, 220:2783, 286:2787, 427:3011, 438:3012, 453:3013, 354:3014, 455:3015, 456:3016}

# seq attention
num_heads = 2
num_blocks = 1
attention_embedding_size = 16
att_user_context_slots = {201, 202, 203, 221}
user_att_feature_names = ['sparse_att_w_' + str(slot_id) for slot_id in att_user_context_slots]
print('user_att_feature_names:', user_att_feature_names)
for slot_id, feature_name in zip(att_user_context_slots, user_att_feature_names):
    print(slot_id, feature_name)

att_doc_slots = {401, 407, 412, 457, 458, 459}
doc_att_feature_names = ['sparse_att_w_' + str(slot_id) for slot_id in att_doc_slots]
print('doc_att_feature_names:', doc_att_feature_names)
for slot_id, feature_name in zip(att_doc_slots, doc_att_feature_names):
    print(slot_id, feature_name)

autogroup_slots = [201, 202, 203, 204, 209, 211, 213, 216, 219, 220, 221, 222, 223, 226, 227, 228,
                   232, 255, 266, 267, 284, 285, 286, 289, 290, 291, 292, 293, 294, 295, 296, 297,
                   298, 1403, 1404, 1711, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614]

autogroup_feature_names = ['sparse_w_' + str(slot_id) for slot_id in autogroup_slots]
print('autogroup_feature_names:', autogroup_feature_names)

ORDERS = [2, 3, 4]
GROUP_NUMS = [20, 10, 5]

# dense embedding
EMBEDDING_DIM = 8

label_configs = [['play_time', 10000, 1, 1],
                 ['video_time', 10011, 0, 0],
                 ['noise_flag', 10000, 8, 8],
                 ['fresh_num', 10012, 0, 0]
                 ]

task_num = 4
expert_num = 5
bins = [15, 30, 46, 93, 125]
bins_fresh = [1.5, 2.5, 3.5, 4.5, 10.5]

use_bn = True


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


class QBMiniFLoatModel(KerasModel):
    def __init__(self, name, configs):
        super().__init__(name, configs)

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
        inputs = self.preprocess_inputs(inputs)
        u_amm = inputs["u_amm"]
        i_amm = inputs["i_amm"]
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
            base_user_layer = user_expert(user_merge_layers,[user_extra_merge_layers, new_add_user_features], training)
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
        for j in range(task_num):
            dnn_net = tf.concat([dnn_gate_doc[j], att_doc_layer, fm_doc_layer], axis=1)
            dnn_out = self.doc_task_towers[j](dnn_net, i_amm, training)
            dnn_multitask_doc.append(dnn_out)

        dnn_multitask_user = []
        for j in range(task_num):
            dnn_net = tf.concat([dnn_gate_user[j], user_attention, fm_user_layer], axis=1)
            dnn_out = self.user_task_towers[j](dnn_net, u_amm, training)
            dnn_multitask_user.append(dnn_out)

        user_vec = tf.concat([dnn_multitask_user[2],
                              dnn_multitask_user[0],
                              dnn_multitask_user[1],
                              dnn_multitask_user[3],], axis=1)
        doc_vec = tf.concat([dnn_multitask_doc[2],
                             dnn_multitask_doc[0],
                             dnn_multitask_doc[1],
                             dnn_multitask_doc[3],], axis=1)

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
                         "u_amm_feat": u_amm,
                         "i_amm_feat": i_amm,
                         "finish_user_vec": dnn_multitask_user[1],
                         "finish_doc_vec": dnn_multitask_doc[1]
                         }

        return model_outputs


class Runner(NumerousModel):
    def __init__(self, name):
        super().__init__(name)
        self.add_metric('auc_skip', numerous.metrics.AUC(name='auc_skip'))
        self.add_metric('auc_finish', numerous.metrics.AUC(name='auc_finish'))
        self.add_metric('ratio_loss', numerous.metrics.MeanLoss(name='ratio_loss_mse'))
        self.add_metric('time_loss', numerous.metrics.MeanLoss(name='time_loss_mse'))
        self.add_metric('final_loss', numerous.metrics.MeanLoss(name='final_loss'))
        self.add_metric('denoise_loss_skip', numerous.metrics.MeanLoss(name='denoise_loss_skip'))
        self.add_metric('denoise_loss_finish', numerous.metrics.MeanLoss(name='denoise_loss_finish'))

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
        inputs.add_sparse_feature(
            feature_name="u_amm",
            slot_id=201,
            embedding_dim=EMBEDDING_DIM * 4,
            initializer=emb_initializer,
            combiner=combiner,
            optimizer=emb_optimizer,
            dump_dtype=numerous.float16
        )
        inputs.add_sparse_feature(
            feature_name="i_amm",
            slot_id=401,
            embedding_dim=EMBEDDING_DIM * 4,
            initializer=emb_initializer,
            combiner=combiner,
            optimizer=emb_optimizer,
            dump_dtype=numerous.float16
        )

        for task, slot_id, lower_bound, upper_bound in label_configs:
            inputs.add_label(label_name="label_{}".format(task), slot_id=slot_id,
                             lower_bound=lower_bound, upper_bound=upper_bound)

        embedding, configs = inputs.build()
        return embedding, configs

    def build_model(self, configs):
        model = QBMiniFLoatModel(name="qbminifloat_model", configs=configs)
        optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=0.0002)
        return model, optimizer

    def compute_loss(self, labels, model_outputs, training):
        playtime = labels['label_play_time']
        video_time = labels['label_video_time']
        noise_flag = labels['label_noise_flag']
        y_data = labels['label_0']
        u_amm = model_outputs["u_amm_feat"]
        i_amm = model_outputs["i_amm_feat"]
        finish_user_vec = model_outputs['finish_user_vec']
        finish_doc_vec = model_outputs['finish_doc_vec']
        y_data = tf.clip_by_value(y_data, 0, 1)

        is_no_noise = tf.greater(noise_flag, 0.5 * tf.ones_like(noise_flag))

        sample_weights = tf.ones_like(playtime, tf.float32)
        sample_weights = tf.where(tf.less(video_time, 5 * tf.ones_like(playtime, tf.float32)),
                                  tf.zeros_like(playtime, tf.float32), sample_weights)  # 物理时长小于5s丢掉
        sample_weights = tf.where(tf.less(playtime, tf.zeros_like(playtime, tf.float32)),
                                  tf.zeros_like(playtime, tf.float32), sample_weights)  # 丢掉播放时长小于0
        sample_weights = tf.where(tf.greater(video_time, 300 * tf.ones_like(playtime, tf.float32)),
                                  tf.zeros_like(playtime, tf.float32), sample_weights)  # 大于250s也不要

        play_ratio = tf.where(tf.less(video_time, tf.ones_like(playtime, tf.float32)),
                              tf.zeros_like(playtime, tf.float32), playtime / video_time)
        is_finish = tf.where(tf.greater_equal(playtime, video_time), tf.ones_like(playtime), tf.zeros_like(playtime))
        is_skip = tf.where(tf.less(playtime, 4.5 * tf.ones_like(playtime)), tf.ones_like(playtime),
                           tf.zeros_like(playtime))
        # ratio
        play_ratio = tf.where(tf.less(video_time, tf.ones_like(playtime, tf.float32)),
                            tf.zeros_like(playtime, tf.float32), playtime / video_time)
        play_time_label = tf.clip_by_value(playtime/110., 0, 1)
        play_ratio_label = tf.clip_by_value(play_ratio, 0, 1)

        playtime = tf.clip_by_value(playtime, 0, 10000)
        video_time = tf.clip_by_value(video_time, 0, 1000)

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
        skip_pred = model_outputs['skip_pred']
        finish_pred = model_outputs['finish_pred']
        ratio_pred = model_outputs['ratio_pred']
        time_pred = model_outputs['time_pred']

        finish_loss_ce = tf.compat.v1.losses.sigmoid_cross_entropy(is_finish, finish_logits, sample_weights_finish,
                                                                   reduction=tf.compat.v1.losses.Reduction.NONE)
        skip_loss_ce = tf.compat.v1.losses.sigmoid_cross_entropy(is_skip, skip_logits, sample_weights_skip,
                                                                 reduction=tf.compat.v1.losses.Reduction.NONE)
        ratio_loss_ce = tf.compat.v1.losses.sigmoid_cross_entropy(play_ratio_label, ratio_logits, sample_weights_finish,
                                                                   reduction=tf.compat.v1.losses.Reduction.NONE)
        time_loss_ce = tf.compat.v1.losses.sigmoid_cross_entropy(play_time_label, time_logits, sample_weights_finish,
                                                                 reduction=tf.compat.v1.losses.Reduction.NONE)
        ratio_loss_mse = tf.compat.v1.losses.mean_squared_error(play_ratio_label, ratio_pred, sample_weights_finish,
                                                                   reduction=tf.compat.v1.losses.Reduction.NONE)
        time_loss_mse = tf.compat.v1.losses.mean_squared_error(play_time_label, time_pred, sample_weights_finish,
                                                                 reduction=tf.compat.v1.losses.Reduction.NONE)
        denoise_loss_skip = tf.compat.v1.losses.sigmoid_cross_entropy(is_skip, denoise_skip_logits,
                                                                      tf.where(is_no_noise,
                                                                               tf.ones_like(y_data) / (tf.reduce_mean(
                                                                                   tf.cast(is_no_noise,
                                                                                           tf.float32)) + 1e-9),
                                                                               tf.zeros_like(y_data)),
                                                                      reduction=tf.compat.v1.losses.Reduction.NONE)
        denoise_loss_finish = tf.compat.v1.losses.sigmoid_cross_entropy(is_finish, denoise_finish_logits,
                                                                        tf.where(is_no_noise,
                                                                                 tf.ones_like(y_data) / (tf.reduce_mean(
                                                                                     tf.cast(is_no_noise,
                                                                                             tf.float32)) + 1e-9),
                                                                                 tf.zeros_like(y_data)),
                                                                        reduction=tf.compat.v1.losses.Reduction.NONE)
        # denoise_loss_skip = tf.compat.v1.losses.sigmoid_cross_entropy(is_skip, denoise_skip_logits,
        #                                                               reduction=tf.compat.v1.losses.Reduction.NONE)
        # denoise_loss_finish = tf.compat.v1.losses.sigmoid_cross_entropy(is_finish, denoise_finish_logits,
        #                                                                 reduction=tf.compat.v1.losses.Reduction.NONE)
        user_amm_loss = tf.reduce_sum(tf.square(tf.stop_gradient(finish_user_vec) - i_amm), axis=-1,
                                      keepdims=True)
        doc_amm_loss = tf.reduce_sum(tf.square(tf.stop_gradient(finish_doc_vec) - u_amm), axis=-1,
                                     keepdims=True)

        amm_loss = 0.5 * tf.multiply(is_finish, user_amm_loss) + 0.5 * tf.multiply(is_finish, doc_amm_loss)
        final_loss = skip_loss_ce + finish_loss_ce + ratio_loss_ce + time_loss_ce + denoise_loss_finish + denoise_loss_skip + amm_loss

        self.get_metric("auc_skip").update_state(is_skip, skip_pred)
        self.get_metric("auc_finish").update_state(is_finish, finish_pred)
        self.get_metric('ratio_loss').update_state(ratio_loss_mse)
        self.get_metric('time_loss').update_state(time_loss_mse)
        self.get_metric("final_loss").update_state(final_loss)
        self.get_metric("denoise_loss_skip").update_state(denoise_loss_skip)
        self.get_metric("denoise_loss_finish").update_state(denoise_loss_finish)

        return final_loss


numerous.cluster.start(numerous.cluster.ClusterConfig(run_mode="cluster"))
with numerous.distribute.get_strategy().scope():
    model = Runner(name='gpu_runner')
    model.run(GPU_CONFIG)
numerous.cluster.stop()
