#!/usr/bin/python3
# coding=utf-8
import sys
import os

sys.path.insert(0, os.getcwd())
import tensorflow as tf
import tensorflow.keras as K

import numerous
from numerous.train.run_config import RunConfig
from model_zoo.inputs import Inputs
from model_zoo.models import NumerousModel
from layers import MultiHeadSelfAttention, SENet, MMOE, get_label, MMoELayer

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
        "sub_dir_file_regexs": [".*part-000.*", ".*part-001.*", ".*part-002.*", ".*part-003.*", ".*part-004.*",
                                ".*part-005.*", ".*part-006.*", ".*part-007.*", ".*part-008.*", ".*part-009.*"],
        "pass_dir_regexs": [".*"],
        "dir_format": "%Y-%m-%d/%H",
        "data_format": "text",
        "sample_col_delimiter": "|",
        "sample_feature_delimiter": ";",
        "feature_section_delimiter": ":",
        "id_col_pos": 0,
        "label_col_pos": 1,
        "feature_col_pos": 2,
        # 特征预处理，没有就不写
        "plugin_path": "./libdefault_sample_preprocessor_trs.so",
        "plugin_config": {},
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
    },
}

data_source = numerous.utils.config.get_config().get_option("data_source", default_value="hdfs")
if data_source == 'cdmq':
    cdmq_interval_min = 20
    GPU_CONFIG['reader_config']['cdmq_consumer_thread_num'] = 8
    GPU_CONFIG['reader_config']['cdmq_interval_min'] = cdmq_interval_min
    GPU_CONFIG['reader_config']['cdmq_max_sample_num_per_pass'] = 500000 * cdmq_interval_min
    GPU_CONFIG['parameter_server_config']['is_serialized_cache'] = True
    GPU_CONFIG['saver_config']['model_dump_interval'] = 3
    GPU_CONFIG['saver_config']['always_complete'] = False

print("GPU_CONFIG:", GPU_CONFIG, flush=True)

# embedding all
all_user_slots = [201, 202, 203, 204, 205, 206, 209, 211, 212, 213, 214, 215, 216, 218, 219, 220, 221, 222, 223, 224,
                  225, 226, 227, 228, 232,
                  233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 252, 253,
                  254, 255, 266, 267, 268,
                  269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 289, 290,
                  291, 292, 293, 294, 295,
                  296, 297, 298,
                  1403, 1404]
user_base_feature_names = ['sparse_usr_w_' + str(slot_id) for slot_id in all_user_slots]

all_doc_slots = [401, 402, 403, 404, 405, 406, 407, 408, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423,
                 424, 425, 426, 427, 428,
                 429, 430, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449,
                 450, 451, 452, 453, 454,
                 455, 456, 457, 458, 459, 460, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
                 494, 495, 496, 497, 498, 499]
doc_base_feature_names = ['sparse_doc_w_' + str(slot_id) for slot_id in all_doc_slots]

num_heads = 2
num_blocks = 1
attention_embedding_size = 16

action_his_vid_slots = list(range(1405, 1465))
action_his_vid_feature_names = ['action_his_vid_w_' + str(slot_id) for slot_id in action_his_vid_slots]
action_his_time_slots = list(range(1465, 1525))
action_his_time_feature_names = ['action_his_time_w_' + str(slot_id) for slot_id in action_his_time_slots]

click_his_vid_slots = list(range(30, 80))
click_his_vid_feature_names = ['click_his_vid_w_' + str(slot_id) for slot_id in click_his_vid_slots]
click_his_pos_slots = list(range(100, 150))
click_his_pos_feature_names = ['click_his_pos_w_' + str(slot_id) for slot_id in click_his_pos_slots]

EMBEDDING_DIM = 16

label_configs = [
    ['play_time', 10000, 1, 1],
    ['itime', 10011, 0, 0],
]

task_names = ['ratio', 'skip', 'is_finish', 'time']
layersizes = [1024, 512, 256, 128]
num_tasks = 4


#################TF2模型结构，通常包含多个layer#########################
class QBMiniFLoatModel(tf.keras.Model):
    def __init__(self, name):
        super().__init__(name=name)
        print("QBMiniFLoatModel super init ok")
        self.action_trans_layer = MultiHeadSelfAttention(attention_embedding_size=attention_embedding_size,
                                                         num_heads=num_heads,
                                                         num_blocks=num_blocks,
                                                         prefix="action")
        self.click_trans_layer = MultiHeadSelfAttention(attention_embedding_size=attention_embedding_size,
                                                        num_heads=num_heads,
                                                        num_blocks=num_blocks,
                                                        prefix="click")
        self.doc_dense_senet = SENet(filed_num=len(all_doc_slots), embedding_dim=EMBEDDING_DIM, squeeze_dim=128)
        self.user_dense_senet = SENet(filed_num=len(all_user_slots), embedding_dim=EMBEDDING_DIM, squeeze_dim=128)
        self.doc_batch_normalization = K.layers.BatchNormalization(name="doc_input_layerbn")
        self.user_batch_normalization = K.layers.BatchNormalization(name="user_input_layerbn")
        # 自定义mmoe
        self.doc_mmoe = MMOE(model_name="doc_mmoe", debug=False)
        self.user_mmoe = MMOE(model_name="user_mmoe", debug=False)
        # modelzoo继承mmoe
        # self.doc_mmoe = MMoELayer(task_names=task_names,
        #                           expert_num=4,
        #                           expert_hidden_sizes=layersizes,
        #                           tower_hidden_sizes=[128, 64, 16])
        # self.user_mmoe = MMoELayer(task_names=task_names,
        #                            expert_num=4,
        #                            expert_hidden_sizes=layersizes,
        #                            tower_hidden_sizes=[128, 64, 16])
        self.doc_layers = []
        self.user_layers = []
        for i in range(0, num_tasks):
            doc_layer1 = K.layers.Dense(units=128, activation='selu', name="doc_layer1%d" % i)
            doc_layer2 = K.layers.Dense(units=64, activation='selu', name="doc_layer2%d" % i)
            doc_layer3 = K.layers.Dense(units=16, activation='selu', name="doc_layer3%d" % i)
            doc_batch_normalization = K.layers.BatchNormalization(name="doc_vec%d" % i)
            self.doc_layers.append((doc_layer1, doc_layer2, doc_layer3, doc_batch_normalization))

            user_layer1 = K.layers.Dense(units=128, activation='selu', name="user_layer1%d" % i)
            user_layer2 = K.layers.Dense(units=64, activation='selu', name="user_layer2%d" % i)
            user_layer3 = K.layers.Dense(units=16, activation='selu', name="user_layer3%d" % i)
            user_batch_normalization = K.layers.BatchNormalization(name="user_vec%d" % i)
            self.user_layers.append((user_layer1, user_layer2, user_layer3, user_batch_normalization))
        print("QBMiniFLoatModel init ok")
        pass

    def call(self, inputs, training):
        print("QBMiniFLoatModel call start")
        action_his_vid_inputs = tf.stack([inputs[feature_name] for feature_name in action_his_vid_feature_names],
                                         axis=1)
        action_his_time_inputs = tf.stack([inputs[feature_name] for feature_name in action_his_time_feature_names],
                                          axis=1)
        action_trans_inputs = tf.concat([action_his_vid_inputs, action_his_time_inputs], axis=-1)
        action_trans_output = self.action_trans_layer(action_trans_inputs, training)

        click_his_vid_inputs = tf.stack([inputs[feature_name] for feature_name in click_his_vid_feature_names], axis=1)
        click_his_pos_inputs = tf.stack([inputs[feature_name] for feature_name in click_his_pos_feature_names], axis=1)
        click_trans_inputs = tf.concat([click_his_vid_inputs, click_his_pos_inputs], axis=-1)
        click_trans_output = self.click_trans_layer(click_trans_inputs, training)

        doc_dense_inputs = tf.stack([inputs[feature_name] for feature_name in doc_base_feature_names], axis=1)
        user_dense_inputs = tf.stack([inputs[feature_name] for feature_name in user_base_feature_names], axis=1)
        print("WTFdocSHAPE:", doc_dense_inputs.get_shape())
        print("WTFusrSHAPE:", user_dense_inputs.get_shape())

        ## squeeze and excitation
        doc_dense_embedding_senet = self.doc_dense_senet(doc_dense_inputs, training)
        user_dense_embedding_senet = self.user_dense_senet(user_dense_inputs, training)

        ## MMOE remove attentions user dual towers
        ## exclude input layer size in layersizes
        ## include two attention inputs

        ## all doc experts
        doc_dense_embedding = tf.concat([doc_dense_embedding_senet], axis=1, name="doc_dense_embedding_final")
        # training
        doc_input = self.doc_batch_normalization(doc_dense_embedding, training)
        tf.print("doc_input: ", doc_input)

        ## all user experts
        user_input = tf.concat([action_trans_output, click_trans_output, user_dense_embedding_senet],
                               axis=1, name="user_input_layer")
        user_input = self.user_batch_normalization(user_input, training)
        tf.print("user_input: ", user_input)

        doc_mmoe = self.doc_mmoe(doc_input, training)
        user_mmoe = self.user_mmoe(user_input, training)
        tf.print("user_mmoe: ", user_mmoe)

        docvecs = []
        uservecs = []
        for i in range(0, num_tasks):
            doc_layer1 = self.doc_layers[i][0](doc_mmoe[i])
            doc_layer2 = self.doc_layers[i][1](doc_layer1)
            doc_layer3 = self.doc_layers[i][2](doc_layer2)
            doc_vecout = self.doc_layers[i][3](doc_layer3, training)
            docvecs.append(doc_vecout)

            user_layer1 = self.user_layers[i][0](user_mmoe[i])
            user_layer2 = self.user_layers[i][1](user_layer1)
            user_layer3 = self.user_layers[i][2](user_layer2)
            user_vecout = self.user_layers[i][3](user_layer3, training)
            uservecs.append(user_vecout)

        user_vec = tf.concat(uservecs, axis=1, name="user_vec")
        tf.print("user_vec: ", user_vec)
        doc_vec = tf.concat(docvecs, axis=1, name="doc_vec")
        tf.print("doc_vec: ", doc_vec)

        # compute loss
        # ratio
        pr_inner_product = tf.multiply(uservecs[0], docvecs[0])
        y_pred1 = tf.sigmoid(tf.reduce_sum(pr_inner_product, 1, keepdims=True), name="pr_Sigmoid")
        tf.print("y_pred1: ", y_pred1)
        # skip
        pr_inner_product2 = tf.reduce_sum(tf.multiply(uservecs[1], docvecs[1]), 1, keepdims=True)
        y_pred2 = tf.sigmoid(pr_inner_product2, name="pr_Sigmoid2")
        # is_finish
        pr_inner_product3 = tf.reduce_sum(tf.multiply(uservecs[2], docvecs[2]), 1, keepdims=True)
        y_pred3 = tf.sigmoid(pr_inner_product3, name="pr_Sigmoid3")
        # time_label
        pr_inner_product4 = tf.multiply(uservecs[3], docvecs[3])
        y_pred4 = tf.sigmoid(tf.reduce_sum(pr_inner_product4, 1, keepdims=True), name="pr_Sigmoid4")

        final_output = tf.concat([y_pred1, y_pred2, y_pred3, y_pred4], axis=1, name="pred_output")

        model_outputs = {
            'user_vec': user_vec,
            'doc_vec': doc_vec,
            'ratio_pred': y_pred1,
            'skip_pred': y_pred2,
            'is_finish_pred': y_pred3,
            'time_pred': y_pred4,
            'skip_logit': pr_inner_product2,
            'is_finish_logit': pr_inner_product3,
            'final_output': final_output
        }
        return model_outputs


class Runner(NumerousModel):
    def __init__(self, name):
        super().__init__(name=name)
        print("Runner super init ok")
        self.model = QBMiniFLoatModel(name='QBMiniFLoatModel')  # define model
        self.set_optimizer(tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, learning_rate=0.00005))  # dense optimizer
        self.add_metric('final_loss', numerous.metrics.Mean(name='final_loss'))
        self.add_metric('ratio_loss', numerous.metrics.Mean(name='ratio_loss'))
        self.add_metric('skip_loss', numerous.metrics.Mean(name='skip_loss'))
        self.add_metric('finish_loss', numerous.metrics.Mean(name='finish_loss'))
        self.add_metric('time_loss', numerous.metrics.Mean(name='time_loss'))
        self.add_metric('auc_ratio', numerous.metrics.AUC(name='auc_ratio'))
        self.add_metric('auc_skip', numerous.metrics.AUC(name='auc_skip'))
        self.add_metric('auc_finish', numerous.metrics.AUC(name='auc_finish'))
        print("Runner init ok")

    def build_inputs(self):
        inputs = Inputs(optimizer=numerous.optimizers.Adam(learning_rate=0.00005))
        emb_initializer = numerous.initializers.RandomUniform(minval=-0.001, maxval=0.001)
        emb_optimizer = numerous.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)

        combiner = numerous.layers.Combiner.SUM
        for slot_id, feature_name in zip(all_doc_slots, doc_base_feature_names):
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.int8
            )
        for slot_id, feature_name in zip(all_user_slots, user_base_feature_names):
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.int8
            )
        for slot_id, feature_name in zip(action_his_vid_slots, action_his_vid_feature_names):
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.int8
            )

        for slot_id, feature_name in zip(action_his_time_slots, action_his_time_feature_names):
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.int8
            )

        for slot_id, feature_name in zip(click_his_vid_slots, click_his_vid_feature_names):
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.int8
            )
        for slot_id, feature_name in zip(click_his_pos_slots, click_his_pos_feature_names):
            inputs.add_sparse_feature(
                feature_name=feature_name,
                slot_id=slot_id,
                embedding_dim=EMBEDDING_DIM,
                initializer=emb_initializer,
                combiner=combiner,
                optimizer=emb_optimizer,
                dump_dtype=numerous.int8
            )

        for task, slot_id, lower_bound, upper_bound in label_configs:
            inputs.add_label(label_name="label_{}".format(task), slot_id=slot_id,
                             lower_bound=lower_bound, upper_bound=upper_bound)
        return inputs

    def call(self, inputs, training):
        inputs = self.preprocess_inputs(inputs)  # process the features into a dictionary, optional
        outputs = self.model(inputs, training)
        return outputs

    def compute_loss(self, labels, model_outputs, weights, training):
        # label_configs = [
        #     ['play_time', 10000, 1, 1],
        #     ['iTime', 10011, 0, 0],
        # ]
        print("compute_loss start")
        playtime = labels['label_play_time']
        playtime = tf.reshape(playtime, shape=[-1, 1])

        iTime = labels['label_itime']
        iTime = tf.reshape(iTime, shape=[-1, 1])

        label = get_label(playtime, iTime)
        # skiplabel
        sPlaytime_shape = tf.fill(tf.shape(playtime), 5.0)
        skip_label = tf.less(playtime, sPlaytime_shape)
        skip_label = tf.cast(skip_label, tf.float32, name="skip_label")
        # isfinishlabel
        sFinish_shape = tf.fill(tf.shape(label), 1.0)
        finishplay_label = tf.greater_equal(label, sFinish_shape)
        finishplay_label = tf.cast(finishplay_label, tf.float32, name="finishplay_label")
        # time label
        time_num = tf.clip_by_value(playtime, 3, 110)
        time_label = time_num / 110.0

        ratio_pred = model_outputs['ratio_pred']
        skip_pred = model_outputs['skip_pred']
        time_pred = model_outputs['time_pred']
        is_finish_pred = model_outputs['is_finish_pred']
        skip_logit = model_outputs['skip_logit']
        is_finish_logit = model_outputs['is_finish_logit']
        # ratio_loss = tf.reduce_mean(tf.math.squared_difference(label, ratio_pred), name="ratio_loss")
        ratio_loss = tf.math.squared_difference(label, ratio_pred)
        # skip_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=skip_logit, labels=skip_label),
        #                            name="skip_loss")
        skip_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=skip_logit, labels=skip_label)
        # finish_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=is_finish_logit, labels=finishplay_label),
        #     name="finishplay_loss")
        finish_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=is_finish_logit, labels=finishplay_label)
        # time_loss = tf.reduce_mean(tf.math.squared_difference(time_label, time_pred), name="time_loss")
        time_loss = tf.math.squared_difference(time_label, time_pred)
        # final_loss = tf.add_n([ratio_loss, 0.09 * finish_loss, 0.14 * skip_loss, time_loss], name="final_loss")
        final_loss = tf.add_n([ratio_loss, 0.09 * finish_loss, 0.14 * skip_loss, time_loss], name="final_loss")

        self.get_metric('final_loss').update_state(final_loss)
        self.get_metric('ratio_loss').update_state(ratio_loss)
        self.get_metric('skip_loss').update_state(skip_loss)
        self.get_metric('finish_loss').update_state(finish_loss)
        self.get_metric('time_loss').update_state(time_loss)
        self.get_metric('auc_ratio').update_state(label, ratio_pred)
        self.get_metric('auc_skip').update_state(skip_label, skip_pred)
        self.get_metric('auc_finish').update_state(finishplay_label, is_finish_pred)

        total_loss = [ratio_loss, skip_loss, finish_loss, time_loss, final_loss]
        return total_loss


numerous.cluster.start(numerous.cluster.get_cluster_config(run_mode=numerous.cluster.RunMode.CLUSTER))
with numerous.distribute.get_strategy().scope():
    print("start init model")
    model = Runner(name="gpu_runner")
    model.init_by_run_config(RunConfig(GPU_CONFIG))
    model.run()
numerous.cluster.stop()
