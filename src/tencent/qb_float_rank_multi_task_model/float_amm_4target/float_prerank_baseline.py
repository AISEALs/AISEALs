#!/usr/bin/python3

import tensorflow as tf
import tensorflow.keras as K

import numerous
from numerous.train.run_config import RunConfig
from numerous.utils.config import get_config
from model_zoo.inputs import Inputs
from model_zoo.models import NumerousModel
from model_zoo.layers import MLP
from layers_new import FM, MultiHeadSelfAttention



is_shapley = int(get_config().get_option("userdefine.config.shapley", default_value="0"))


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
        "is_ignore_non_batch_ins": True,
        "file_loader_thread_num": 32 * GPU_CARD_NUM,
        "extract_unique_key_thread_num": 32 * GPU_CARD_NUM,
        "merge_thread_num": 8 * GPU_CARD_NUM,
        "write_fileins_thread_num": 4 * GPU_CARD_NUM,
        "write_unique_key_thread_num": 4 * GPU_CARD_NUM,
        "sub_dir_file_regexs": [".*part-000.*", ".*part-001.*", ".*part-002.*", ".*part-003.*", ".*part-004.*"],
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
        "is_ignore_model_data_time" : True, # False 从模型训练时间开始，True从配置时间开始

    },
    "parameter_server_config": {
        "part_num": 1024,
        "hbm_max_key_num": 1024 * 1024 * 32,
        "hbm_max_dynamic_byte_size": 1024 * 1024 * 1024 * 8,
        "hbm_hashmap_load_factor": 0.75,
        "prepare_pass_thread_num": 12 * GPU_CARD_NUM,
        "build_pass_thread_num": 12 * GPU_CARD_NUM,
        "build_group_thread_num": 12 * GPU_CARD_NUM,
        "build_group_postprocess_thread_num": 4 * GPU_CARD_NUM,
        "do_persistence_thread_num": 24 * GPU_CARD_NUM,
        "load_checkpoint_thread_num": 24 * GPU_CARD_NUM,
        "use_parallel_optimizer": False,
        "delete_after_unseen_days": 7,
        "cow_clip": {"r": 1.0, "lower_bound": 1e-5},
        "do_recover_thread_num": 256,
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
all_user_slots = [266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
                  286, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 1711, 201, 202, 203, 204, 205, 206, 209, 211,
                  212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 232, 233, 234, 235,
                  236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 252, 253, 254, 255]
user_base_feature_names = ['sparse_w_' + str(slot_id) for slot_id in all_user_slots]

all_doc_slots = [401, 402, 403, 404, 405, 406, 407, 408, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423,
                 424, 425, 426, 427, 428, 429, 430, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444,
                 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460]
doc_base_feature_names = ['sparse_w_' + str(slot_id) for slot_id in all_doc_slots]

# fm embedding sum
doc_fm_slots = [401, 402, 403, 404, 405, 406, 407, 408, 412, 413, 414, 415, 416, 417, 418, 419, 420, 435, 457, 458, 459,
                460]
doc_fm_feature_names = ['sparse_fm_w_' + str(slot_id) for slot_id in doc_fm_slots]

user_fm_slots = [266, 267, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 202, 203, 204, 205, 206, 209, 211, 212,
                 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 232, 233, 234, 235, 236,
                 237, 238, 239, 240, 255]
user_fm_feature_names = ['sparse_fm_w_' + str(slot_id) for slot_id in user_fm_slots]

# seq
seq_item_slots = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                  55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
seq_item_feature_names = ['sparse_w_' + str(slot_id) for slot_id in seq_item_slots]

seq_pos_slots = [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
                 148, 149, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
                 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
seq_pos_feature_names = ['sparse_w_' + str(slot_id) for slot_id in seq_pos_slots]

# seq attention
num_heads = 2
num_blocks = 1
attention_embedding_size = 16
att_user_context_slots = [201, 202, 203, 221]
user_att_feature_names = ['sparse_att_w_' + str(slot_id) for slot_id in att_user_context_slots]

att_doc_slots = [457, 458, 459, 401, 407, 412]
doc_att_feature_names = ['sparse_att_w_' + str(slot_id) for slot_id in att_doc_slots]

# dense embedding
EMBEDDING_DIM = 8

label_configs = [['play_time', 10000, 1, 1],
                 ['video_time', 10011, 0, 0],
                 ['noise_flag', 10000, 8, 8],
                 ['fresh_num', 10012, 0, 0]
                 ]

task_num = 2
expert_num = 5
bins = [15, 30, 46, 93, 125]
bins_fresh = [1.5, 2.5, 3.5, 4.5, 10.5]

use_bn = True

def dump_tensor_filter(pred_tensors):
    if len(pred_tensors) >= 0:
        dump_tensors = {
            'shapley_sample_info:0':pred_tensors['shapley_sample_info'],
            # 'batch_skip_weight_loss:0':pred_tensors['batch_skip_weight_loss'],
            # 'batch_finish_weight_loss:0':pred_tensors['batch_finish_weight_loss'],
            'batch_skip_loss:0':pred_tensors['batch_skip_loss'],
            'batch_finish_loss:0':pred_tensors['batch_finish_loss'],
            # 'skip_logits':pred_tensors['skip_logits'],
            # 'user_vec_infer':pred_tensors['user_vec_infer'],
            # 'doc_vec_infer':pred_tensors['doc_vec_infer']
        } # Dict
        return dump_tensors
    else:
        raise ValueError("call function return empty object")


class DenoiseLayer(K.layers.Layer):
    def __init__(self, **kwargs):
        super(DenoiseLayer, self).__init__(**kwargs)
        self.mlp = MLP(hidden_units=[512, 256, 32, 1], use_bn=use_bn)
        self.model = None

    def call(self, inputs, training):
        user_embedding, doc_embedding = inputs
        feature_input = tf.stop_gradient(tf.concat([user_embedding, doc_embedding], axis=1))
        denoise_logits = self.mlp(feature_input, training=training)
        self.model = denoise_logits
        return self.model


class QBMiniFLoatModel(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)

        self.doc_experts = []
        for i in range(expert_num):
            expert = MLP(hidden_units=[512, 256, 128], use_bn=use_bn, name=f"base_doc_expert_{i}")
            self.doc_experts.append(expert)
        self.doc_gates = []
        for j in range(task_num):
            gate = K.layers.Dense(units=expert_num, activation='softmax', name=f"base_doc_gate_{j}")
            self.doc_gates.append(gate)

        self.user_experts = []
        for i in range(expert_num):
            expert = MLP(hidden_units=[512, 256, 128], use_bn=use_bn, name=f"base_user_expert_{i}")
            self.user_experts.append(expert)
        self.user_gates = []
        for j in range(task_num):
            gate = K.layers.Dense(units=expert_num, activation='softmax', name=f"base_user_gate_{j}")
            self.user_gates.append(gate)

        self.doc_fm = FM(name="doc_fm")
        self.user_fm = FM(name="user_fm")

        self.fm_doc_mlp = MLP(hidden_units=[32, 32], use_bn=use_bn, name="fm_doc")
        self.fm_user_mlp = MLP(hidden_units=[32, 32], use_bn=use_bn, name="fm_user")

        self.user_att_layer = MultiHeadSelfAttention(attention_embedding_size=attention_embedding_size,
                                                     num_heads=num_heads,
                                                     num_blocks=num_blocks)

        self.user_attention_mlp = MLP(hidden_units=[32], name="user_attention")
        self.doc_attention_mlp = MLP(hidden_units=[32, 32], use_bn=use_bn, name="doc_attention")

        self.doc_task_towers = []
        for j in range(task_num):
            tower = MLP(hidden_units=[128, 64, 32], use_bn=use_bn, name=f"doc_tower_{j}")
            self.doc_task_towers.append(tower)

        self.user_task_towers = []
        for j in range(task_num):
            tower = MLP(hidden_units=[128, 64, 32], use_bn=use_bn, name=f"user_tower_{j}")
            self.user_task_towers.append(tower)

        self.denoise_net_finish = DenoiseLayer(name="denoise_net_finish")
        self.denoise_net_skip = DenoiseLayer(name="denoise_net_skip")

    def call(self, inputs, training):
        # doc mmoe
        doc_merge_layers = tf.concat([inputs[feature_name] for feature_name in doc_base_feature_names], axis=1)
        expert_net_doc = []
        for doc_expert in self.doc_experts:
            base_doc_layer = doc_expert(doc_merge_layers, training)
            expert_net_doc.append(base_doc_layer)
        expert_net_doc = tf.stack(expert_net_doc)
        expert_net_doc = tf.transpose(expert_net_doc, perm=[0, 2, 1])
        dnn_gate_doc = []
        for doc_gate in self.doc_gates:
            expert_weight = doc_gate(doc_merge_layers)
            gate_net = tf.einsum('ai,ija->aj', expert_weight, expert_net_doc)
            dnn_gate_doc.append(gate_net)

        # user mmoe
        expert_net_user = []
        user_merge_layers = tf.concat([inputs[feature_name] for feature_name in user_base_feature_names], axis=1)

        for user_expert in self.user_experts:
            base_user_layer = user_expert(user_merge_layers, training)
            expert_net_user.append(base_user_layer)
        expert_net_user = tf.stack(expert_net_user)
        expert_net_user = tf.transpose(expert_net_user, perm=[0, 2, 1])
        dnn_gate_user = []
        for user_gate in self.user_gates:
            expert_weight = user_gate(user_merge_layers)
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
            dnn_out = self.doc_task_towers[j](dnn_net, training)
            dnn_multitask_doc.append(dnn_out)

        dnn_multitask_user = []
        for j in range(task_num):
            dnn_net = tf.concat([dnn_gate_user[j], user_attention, fm_user_layer], axis=1)
            dnn_out = self.user_task_towers[j](dnn_net, training)
            dnn_multitask_user.append(dnn_out)

        user_vec = tf.concat(dnn_multitask_user, axis=1)
        doc_vec = tf.concat(dnn_multitask_doc, axis=1)

        user_vec = tf.identity(user_vec, name="user_vec_infer")
        doc_vec = tf.identity(doc_vec, name="doc_vec_infer")

        inner_product = tf.multiply(dnn_multitask_user[0], dnn_multitask_doc[0])
        skip_logits = tf.reduce_sum(inner_product, 1, keepdims=True)
        skip_pred = tf.sigmoid(skip_logits, name='skip_pred')

        inner_product = tf.multiply(dnn_multitask_user[1], dnn_multitask_doc[1])
        finish_logits = tf.reduce_sum(inner_product, 1, keepdims=True)
        finish_pred = tf.sigmoid(finish_logits, name='finish_pred')

        denoise_finish_logits = self.denoise_net_finish([user_merge_layers, doc_merge_layers], training)
        denoise_skip_logits = self.denoise_net_skip([user_merge_layers, doc_merge_layers], training)

        model_outputs = {'user_vec_infer': user_vec,
                         'doc_vec_infer': doc_vec,
                         'denoise_finish_logits': denoise_finish_logits,
                         'denoise_skip_logits': denoise_skip_logits,
                         "skip_logits": skip_logits,
                         'skip_pred': skip_pred,
                         'finish_logits': finish_logits,
                         'finish_pred': finish_pred
                         }

        return model_outputs

class Runner(NumerousModel):
    def __init__(self, name):
        super().__init__(name=name)
        self.model = QBMiniFLoatModel(name='qbminifloat_model')  # define model
        self.set_optimizer(tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=0.0002))  # dense optimizer
        self.add_metric('auc_skip', numerous.metrics.AUC(name='auc_skip'))
        self.add_metric('auc_finish', numerous.metrics.AUC(name='auc_finish'))
        self.add_metric('final_loss', numerous.metrics.Mean(name='final_loss'))
        self.add_metric('denoise_loss_skip', numerous.metrics.Mean(name='denoise_loss_skip'))
        self.add_metric('denoise_loss_finish', numerous.metrics.Mean(name='denoise_loss_finish'))

        if is_shapley > 0:
            # Create tensor dumper by coding
            TENSOR_DUMPER_CONFIG = {
                "type": "hdfs",
                "hdfs": {
                    "hdfs_host": "mdfs://cloudhdfs/mttsparknew",
                    "path": "data/video/rickycui/shapley/floatvideoprerank/predict"
                }
            }

            tensor_dumper_storage = numerous.train.saver.get_storage(TENSOR_DUMPER_CONFIG)
            tensor_dumper_config = numerous.train.tensor_dumper.TensorDumperConfig(
                buffer_size=0,
                thread_num=4,
                storage=tensor_dumper_storage,
                dump_format=numerous.train.tensor_dumper.TensorDumperFormat.PROTOBUF_TEXT)
            tensor_dumper = numerous.train.tensor_dumper.TensorDumper(tensor_dumper_config)
            # if not set filter, will dump nothing
            tensor_dumper.set_filter_fn(dump_tensor_filter)
            # demo hook of deploying method
            TENSOR_DUMPER_NAME = "mlp_dump_tensor"
            self.set_tensor_dumper(tensor_dumper, name=TENSOR_DUMPER_NAME)


    def build_inputs(self):
        inputs = Inputs(optimizer=numerous.optimizers.Adam(learning_rate=0.001))
        emb_initializer = numerous.initializers.RandomUniform(minval=-0.001, maxval=0.001)
        emb_optimizer = numerous.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        combiner = numerous.layers.Combiner.SUM
        for slot_id, feature_name in zip(all_doc_slots, doc_base_feature_names):
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
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.float16
            )

        for task, slot_id, lower_bound, upper_bound in label_configs:
            # inputs.add_label(label_name="label_{}".format(task), slot_id=slot_id,
            #                  lower_bound=lower_bound, upper_bound=upper_bound)
            inputs.add_dense_feature("label_{}".format(task), slot_id=slot_id,
                                     lower_bound=lower_bound, upper_bound=upper_bound)

        if is_shapley > 0:
            inputs.add_dense_feature("shapley_sample_info_input", slot_id=99999989, lower_bound=1, upper_bound=6)
        return inputs

    def call(self, inputs, training):
        inputs = self.preprocess_inputs(inputs)  # process the features into a dictionary, optional
        outputs = self.model(inputs, training)
        if is_shapley > 0:
            outputs['shapley_sample_info'] = inputs['shapley_sample_info_input']
        self.compute_loss_core(inputs, outputs)

        return outputs

    def compute_loss_core(self, inputs,model_outputs):
        playtime = inputs['label_play_time']
        video_time = inputs['label_video_time']
        noise_flag = inputs['label_noise_flag']
        y_data = inputs['label']

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


        finish_loss_ce = tf.compat.v1.losses.sigmoid_cross_entropy(is_finish, finish_logits, sample_weights_finish,
                                                                   reduction=tf.compat.v1.losses.Reduction.NONE)
        skip_loss_ce = tf.compat.v1.losses.sigmoid_cross_entropy(is_skip, skip_logits, sample_weights_skip,
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

        final_loss = skip_loss_ce + finish_loss_ce + denoise_loss_finish + denoise_loss_skip


        batch_finish_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=finish_logits, labels=is_finish)
        batch_skip_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=skip_logits, labels=is_skip)


        model_outputs['batch_finish_weight_loss'] = finish_loss_ce
        model_outputs['batch_skip_weight_loss'] = skip_loss_ce
        model_outputs['batch_finish_loss'] = batch_finish_loss
        model_outputs['batch_skip_loss'] = batch_skip_loss

        model_outputs['sample_weights_skip'] = sample_weights_skip


        model_outputs['denoise_loss_skip'] = denoise_loss_skip
        model_outputs['denoise_loss_finish'] = denoise_loss_finish

        model_outputs['is_finish'] = is_finish
        model_outputs['is_skip'] = is_skip

        model_outputs['final_loss'] = final_loss


    def compute_loss(self, labels, model_outputs, weights, training):
        final_loss = model_outputs['final_loss']
        denoise_loss_skip = model_outputs['denoise_loss_skip']
        denoise_loss_finish = model_outputs['denoise_loss_finish']
        is_finish = model_outputs['is_finish']
        is_skip = model_outputs['is_skip']
        skip_pred = model_outputs['skip_pred']
        finish_pred = model_outputs['finish_pred']

        self.get_metric("auc_skip").update_state(is_skip, skip_pred)
        self.get_metric("auc_finish").update_state(is_finish, finish_pred)
        self.get_metric("final_loss").update_state(final_loss)
        self.get_metric("denoise_loss_skip").update_state(denoise_loss_skip)
        self.get_metric("denoise_loss_finish").update_state(denoise_loss_finish)
        return final_loss


numerous.cluster.start(numerous.cluster.get_cluster_config(run_mode=numerous.cluster.RunMode.CLUSTER))
with numerous.distribute.get_strategy().scope():
    model = Runner(name="gpu_runner")
    model.init_by_run_config(RunConfig(GPU_CONFIG))
    model.run()
numerous.cluster.stop()
