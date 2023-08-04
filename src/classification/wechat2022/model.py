import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from category_id_map import CATEGORY_ID_LIST
from network.focal_loss import FocalLoss

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from network.multi_modal_bert import MultiModalBertForMaskedLM
from network.senet import SENet
from network.nextvlad import NeXtVLAD
from network.mean_pooling import MeanPooling


BERT_WEIGHTS_NAME = 'pytorch_model.bin'

'''
class MultiModalV3(Module):
    def __init__(self, config):

        super(ResNetVLBERTForAttentionVis, self).__init__(config)

        self.image_feature_extractor = FastRCNN(config,
                                                average_pool=True,
                                                final_dim=config.NETWORK.IMAGE_FINAL_DIM,
                                                enable_cnn_reg_loss=False)
        self.object_linguistic_embeddings = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        if config.NETWORK.IMAGE_FEAT_PRECOMPUTED or (not config.NETWORK.MASK_RAW_PIXELS):
            self.object_mask_visual_embedding = nn.Embedding(1, 2048)
        if config.NETWORK.WITH_MVRC_LOSS:
            self.object_mask_word_embedding = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.aux_text_visual_embedding = nn.Embedding(1, config.NETWORK.VLBERT.hidden_size)
        self.image_feature_bn_eval = config.NETWORK.IMAGE_FROZEN_BN
        self.tokenizer = BertTokenizer.from_pretrained(config.NETWORK.BERT_MODEL_NAME)
        language_pretrained_model_path = None
        if config.NETWORK.BERT_PRETRAINED != '':
            language_pretrained_model_path = '{}-{:04d}.model'.format(config.NETWORK.BERT_PRETRAINED,
                                                                      config.NETWORK.BERT_PRETRAINED_EPOCH)
        elif os.path.isdir(config.NETWORK.BERT_MODEL_NAME):
            weight_path = os.path.join(config.NETWORK.BERT_MODEL_NAME, BERT_WEIGHTS_NAME)
            if os.path.isfile(weight_path):
                language_pretrained_model_path = weight_path

        if language_pretrained_model_path is None:
            print("Warning: no pretrained language model found, training from scratch!!!")

        self.vlbert = VisualLinguisticBert(
            config.NETWORK.VLBERT,
            language_pretrained_model_path=None if config.NETWORK.VLBERT.from_scratch else language_pretrained_model_path
        )

        # init weights
        self.init_weight()

        self.fix_params()

    def init_weight(self):
        if self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED or (not self.config.NETWORK.MASK_RAW_PIXELS):
            self.object_mask_visual_embedding.weight.data.fill_(0.0)
        if self.config.NETWORK.WITH_MVRC_LOSS:
            self.object_mask_word_embedding.weight.data.normal_(mean=0.0, std=self.config.NETWORK.VLBERT.initializer_range)
        self.aux_text_visual_embedding.weight.data.normal_(mean=0.0, std=self.config.NETWORK.VLBERT.initializer_range)
        self.image_feature_extractor.init_weight()
        if self.object_linguistic_embeddings is not None:
            self.object_linguistic_embeddings.weight.data.normal_(mean=0.0,
                                                                  std=self.config.NETWORK.VLBERT.initializer_range)

    def train(self, mode=True):
        super(ResNetVLBERTForAttentionVis, self).train(mode)
        # turn some frozen layers to eval mode
        if self.image_feature_bn_eval:
            self.image_feature_extractor.bn_eval()

    def fix_params(self):
        pass

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def forward(self,
                image,
                boxes,
                im_info,
                text,
                relationship_label,
                mlm_labels,
                mvrc_ops,
                mvrc_labels,
                *aux):

        # concat aux texts from different dataset
        # assert len(aux) > 0 and len(aux) % 2 == 0
        aux_text_list = aux[0::2]
        aux_text_mlm_labels_list = aux[1::2]
        num_aux_text = sum([_text.shape[0] for _text in aux_text_list])
        max_aux_text_len = max([_text.shape[1] for _text in aux_text_list]) if len(aux_text_list) > 0 else 0
        aux_text = text.new_zeros((num_aux_text, max_aux_text_len))
        aux_text_mlm_labels = mlm_labels.new_zeros((num_aux_text, max_aux_text_len)).fill_(-1)
        _cur = 0
        for _text, _mlm_labels in zip(aux_text_list, aux_text_mlm_labels_list):
            _num = _text.shape[0]
            aux_text[_cur:(_cur + _num), :_text.shape[1]] = _text
            aux_text_mlm_labels[_cur:(_cur + _num), :_text.shape[1]] = _mlm_labels
            _cur += _num

        ###########################################

        # visual feature extraction
        images = image
        box_mask = (boxes[:, :, 0] > -1.5)
        origin_len = boxes.shape[1]
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        mvrc_ops = mvrc_ops[:, :max_len]
        mvrc_labels = mvrc_labels[:, :max_len]

        if self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED:
            box_features = boxes[:, :, 4:]
            box_features[mvrc_ops == 1] = self.object_mask_visual_embedding.weight[0]
            boxes[:, :, 4:] = box_features

        obj_reps = self.image_feature_extractor(images=images,
                                                boxes=boxes,
                                                box_mask=box_mask,
                                                im_info=im_info,
                                                classes=None,
                                                segms=None,
                                                mvrc_ops=mvrc_ops,
                                                mask_visual_embed=self.object_mask_visual_embedding.weight[0]
                                                if (not self.config.NETWORK.IMAGE_FEAT_PRECOMPUTED)
                                                   and (not self.config.NETWORK.MASK_RAW_PIXELS)
                                                else None)

        ############################################

        # prepare text
        text_input_ids = text
        text_tags = text.new_zeros(text.shape)
        text_visual_embeddings = self._collect_obj_reps(text_tags, obj_reps['obj_reps'])

        object_linguistic_embeddings = self.object_linguistic_embeddings(
            boxes.new_zeros((boxes.shape[0], boxes.shape[1])).long()
        )
        if self.config.NETWORK.WITH_MVRC_LOSS:
            object_linguistic_embeddings[mvrc_ops == 1] = self.object_mask_word_embedding.weight[0]
        object_vl_embeddings = torch.cat((obj_reps['obj_reps'], object_linguistic_embeddings), -1)

        # add auxiliary text
        max_text_len = max(text_input_ids.shape[1], aux_text.shape[1])
        text_input_ids_multi = text_input_ids.new_zeros((text_input_ids.shape[0] + aux_text.shape[0], max_text_len))
        text_input_ids_multi[:text_input_ids.shape[0], :text_input_ids.shape[1]] = text_input_ids
        text_input_ids_multi[text_input_ids.shape[0]:, :aux_text.shape[1]] = aux_text
        text_token_type_ids_multi = text_input_ids_multi.new_zeros(text_input_ids_multi.shape)
        text_mask_multi = (text_input_ids_multi > 0)
        text_visual_embeddings_multi = text_visual_embeddings.new_zeros((text_input_ids.shape[0] + aux_text.shape[0],
                                                                         max_text_len,
                                                                         text_visual_embeddings.shape[-1]))
        text_visual_embeddings_multi[:text_visual_embeddings.shape[0], :text_visual_embeddings.shape[1]] \
            = text_visual_embeddings
        text_visual_embeddings_multi[text_visual_embeddings.shape[0]:] = self.aux_text_visual_embedding.weight[0]
        object_vl_embeddings_multi = object_vl_embeddings.new_zeros((text_input_ids.shape[0] + aux_text.shape[0],
                                                                     *object_vl_embeddings.shape[1:]))
        object_vl_embeddings_multi[:object_vl_embeddings.shape[0]] = object_vl_embeddings
        box_mask_multi = box_mask.new_zeros((text_input_ids.shape[0] + aux_text.shape[0], *box_mask.shape[1:]))
        box_mask_multi[:box_mask.shape[0]] = box_mask


        ###########################################

        # Visual Linguistic BERT

        encoder_layers, _, attention_probs = self.vlbert(text_input_ids_multi,
                                                         text_token_type_ids_multi,
                                                         text_visual_embeddings_multi,
                                                         text_mask_multi,
                                                         object_vl_embeddings_multi,
                                                         box_mask_multi,
                                                         output_all_encoded_layers=True,
                                                         output_attention_probs=True)
        hidden_states = torch.stack(encoder_layers, dim=0).transpose(0, 1).contiguous()
        attention_probs = torch.stack(attention_probs, dim=0).transpose(0, 1).contiguous()

        return {'attention_probs': attention_probs,
                'hidden_states': hidden_states}
'''

class MultiModalV2(nn.Module):
    def __init__(self, config, task=['cate'], init_from_pretrain=True):
        super().__init__()
        self.config = config
        # 得到bert配置
        bert_cfg = BertConfig.from_pretrained('{}/config.json'.format(config.bert_dir))
        #bert_cfg.num_hidden_layers = 1
        self.pooler = MeanPooling()
        
        self.task = set(task)
        # 最终的类别预测分类任务
        if 'cate' in task:
            self.newfc_cate = torch.nn.Linear(bert_cfg.hidden_size, len(CATEGORY_ID_LIST))
        
        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=config.bert_dir)
            self.num_class = cfg['NUM_CLASSES']
            self.vocab_size = bert_cfg.vocab_size
        
        if 'mfm' in task:
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(bert_cfg) 
            
        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(bert_cfg.hidden_size, 1) 

        if init_from_pretrain:
            self.roberta = MultiModalBertForMaskedLM.from_pretrained(config.bert_dir, config=bert_cfg, file_config=config)
        else:
            self.roberta = MultiModalBertForMaskedLM(bert_cfg, config)

    def forward(self, inputs, inference=False, task=None):
        text_input_ids = inputs['doc_input']
        text_mask = inputs['doc_mask']
        video_feature = inputs['frame_input']
        video_mask = inputs['frame_mask']

        res = {}
        
        if task is None:
            sample_task = self.task
        elif type(task) == str:
            sample_task = [task]
        elif type(task) == list:
            sample_task = task
        
        # perpare for pretrain task [mask and get task label]: {'mlm': mask title token} {'mfm': mask frame} {'itm': shuffle title-video}
        return_mlm = False
        if 'mlm' in sample_task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device) # [SEP] 卡 MASK 大师 [SEP]
            return_mlm = True
            
        if 'mfm' in sample_task:
            vm_input = video_feature
            input_feature, video_label = self.vm.torch_mask_frames(video_feature.cpu(), video_mask.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_label = video_label.to(video_feature.device)
            
        if 'itm' in sample_task:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)
            
        # concat features
        features, lm_prediction_scores, mask = self.roberta(video_feature, video_mask, text_input_ids, text_mask, return_mlm=return_mlm)

        # compute pretrain task loss
        if 'mlm' in sample_task:
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1))
            loss += masked_lm_loss / 1.25 / len(sample_task)
            
        if 'mfm' in sample_task:
            vm_output = self.roberta_mvm_lm_header(features[:, 1:video_feature.size()[1] + 1, :])
            masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input, 
                                                     video_mask, video_label, normalize=False)
            loss += masked_vm_loss / 3 / len(sample_task)
            
        if 'itm' in sample_task:
            pred = self.newfc_itm(features[:, 0, :])
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
            loss += itm_loss / 100 / len(sample_task)
        
        if 'cate' in sample_task:
            if self.config.use_bert_mean_pool == 1:
                features_mean = torch.mean(features, 1)
                cate_logits = self.newfc_cate(features_mean)
            elif self.config.use_bert_mean_pool == 2:
                features_mean = self.pooler(features, mask)
                cate_logits = self.newfc_cate(features_mean)
            else:
                cate_logits = self.newfc_cate(torch.relu(features[:, 0, :]))    # [CLS] token对应的embedding送入cate分类head预测cate
            cate_pred_label = torch.argmax(cate_logits, dim=1)
            res['cate_logits'] = cate_logits
            res['cate_pred_label'] = cate_pred_label
            if 'label' in inputs:
                cate_label = inputs['label']
                cate_loss, cate_accuracy, cate_pred_label, cate_label = self.cal_loss(cate_logits, cate_label)
                res['cate_loss'] = cate_loss
                res['cate_accuracy'] = cate_accuracy
                res['cate_label'] = cate_label

        return res

    def cal_loss(self, prediction, label):
        label = label.squeeze(dim=1)
        if self.config.use_focal_loss:
            loss = FocalLoss(len(CATEGORY_ID_LIST))(prediction, label)
        else:
            loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    
    # calc mfm loss 
    def calculate_mfm_loss(self, video_feature_output, video_feature_input, 
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

        video_tr = video_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss

class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        bert_output_size = 768
        self.fusion = ConcatDenseSE(args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
        self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))

    def forward(self, inputs, inference=False):
        # bert_embedding = self.bert(inputs['title_input'], inputs['title_mask'])['pooler_output']
        # asr_embedding = self.bert(inputs['asr_input'], inputs['asr_mask'])['pooler_output']
        # ocr_embedding = self.bert(inputs['ocr_input'], inputs['ocr_mask'])['pooler_output']
        bert_embedding = self.bert(inputs['doc_input'], inputs['doc_mask'])['pooler_output']

        vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
        vision_embedding = self.enhance(vision_embedding)

        final_embedding = self.fusion([vision_embedding, bert_embedding])
        prediction = self.classifier(final_embedding)

        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding
