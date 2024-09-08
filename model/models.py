import math
import pickle
import torch as th
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer, XLNetPreTrainedModel
from transformers.models.xlnet.modeling_xlnet import XLNetLayer, XLNetModel
from torch.nn.parameter import Parameter


class XLNetAug4Mix(XLNetPreTrainedModel):
    def __init__(self, config):
        super(XLNetAug4Mix, self).__init__(config)
        self.layer = nn.ModuleList([XLNetLayer(config) for _ in range(config.num_hidden_layers)])
        self.tokenizer = AutoTokenizer.from_pretrained('./pre_model/xlnet_base_cased')
        self.word_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.attn_type = config.attn_type
        self.dropout = nn.Dropout(config.dropout)
        self.XLNetModel = XLNetModel(config)
        self.relative_positional_encoding = self.XLNetModel.relative_positional_encoding
        self.n_layer = config.n_layer

    def handle_input(self, input_ids, attention_mask):
        # token -> embedding
        inputs_embeds = self.word_embedding(input_ids)
        inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
        qlen, bsz = inputs_embeds.shape[0], inputs_embeds.shape[1]

        attention_mask = attention_mask.transpose(0, 1).contiguous()

        mlen = 0
        klen = mlen + qlen

        dtype_float = self.dtype
        device = self.device

        input_mask = 1.0 - attention_mask
        data_mask = input_mask[None]
        attn_mask = data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = (attn_mask > 0).to(dtype_float)

        if attn_mask is not None:
            non_tgt_mask = -th.eye(qlen).to(attn_mask)
            if mlen > 0:
                non_tgt_mask = th.cat([th.zeros([qlen, mlen]).to(attn_mask), non_tgt_mask], dim=-1)
            non_tgt_mask = ((attn_mask + non_tgt_mask[:, :, None, None]) > 0).to(attn_mask)
        else:
            non_tgt_mask = None

        word_emb_k = inputs_embeds
        output_h = self.dropout(word_emb_k)
        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)
        pos_emb = pos_emb.to(output_h.device)
        pos_emb = self.dropout(pos_emb)

        return output_h, non_tgt_mask, attn_mask, pos_emb

    def forward(
            self,
            hidden_states,
            hidden_states1=None,
            hidden_states2=None,
            lam=None,
            ws=None,
            mix_layer=None,
            attention_mask=None,
            attention_mask1=None,
            attention_mask2=None,
    ):

        output_h, non_tgt_mask, attn_mask, pos_emb = self.handle_input(hidden_states, attention_mask)
        if hidden_states1 is not None:
            output_h1, non_tgt_mask1, attn_mask1, pos_emb1 = self.handle_input(hidden_states1, attention_mask1)
            output_h2, non_tgt_mask2, attn_mask2, pos_emb2 = self.handle_input(hidden_states2, attention_mask2)

        output_g = None
        for i, layer_module in enumerate(self.layer):
            if hidden_states1 is not None:
                if i < mix_layer:
                    outputs = layer_module(
                        output_h,
                        output_g,
                        attn_mask_h=non_tgt_mask,
                        attn_mask_g=attn_mask,
                        r=pos_emb,
                        seg_mat=None,
                        mems=None,
                        head_mask=None,
                        output_attentions=None,
                    )
                    output_h, output_g = outputs[:2]
                    outputs1 = layer_module(
                        output_h1,
                        output_g,
                        attn_mask_h=non_tgt_mask1,
                        attn_mask_g=attn_mask1,
                        r=pos_emb1,
                        seg_mat=None,
                        mems=None,
                        head_mask=None,
                        output_attentions=None,
                    )
                    output_h1, output_g = outputs1[:2]
                    outputs2 = layer_module(
                        output_h,
                        output_g,
                        attn_mask_h=non_tgt_mask,
                        attn_mask_g=attn_mask,
                        r=pos_emb,
                        seg_mat=None,
                        mems=None,
                        head_mask=None,
                        output_attentions=None,
                    )
                    output_h2, output_g = outputs2[:2]

                if i == mix_layer:
                    mix_aug = ws[0] * output_h1 + ws[1] * output_h2
                    # 新隐藏层
                    output_h = lam * output_h + (1 - lam) * mix_aug

                if i > mix_layer:
                    outputs = layer_module(
                        output_h,
                        output_g,
                        attn_mask_h=non_tgt_mask,
                        attn_mask_g=attn_mask,
                        r=pos_emb,
                        seg_mat=None,
                        mems=None,
                        head_mask=None,
                        output_attentions=None,
                    )
                    output_h, output_g = outputs[:2]

            else:
                outputs = layer_module(
                    output_h,
                    output_g,
                    attn_mask_h=non_tgt_mask,
                    attn_mask_g=attn_mask,
                    r=pos_emb,
                    seg_mat=None,
                    mems=None,
                    head_mask=None,
                    output_attentions=None,
                )
                output_h, output_g = outputs[:2]

        output = self.dropout(output_g if output_g is not None else output_h)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = output.permute(1, 0, 2).contiguous()
        return output


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(th.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = th.spmm(adj, input)
        if self.variant:
            support = th.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * th.mm(support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output


class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha, i + 1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner


class XLG_Net(th.nn.Module):
    def __init__(self, pretrained_model='xlnet_base_cased', nb_class=2, m=0.6, n_layer=64, n_hidden=32,
                 dropout=0.5, lamda=0.5, alpha=0.1, variant=False):
        super(XLG_Net, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.xlnet_model = XLNetAug4Mix.from_pretrained(pretrained_model)
        self.feat_dim = self.xlnet_model.config.hidden_size
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcnii = GCNII(
            nfeat=self.feat_dim,
            nlayers=n_layer,
            nhidden=n_hidden,
            nclass=nb_class,
            dropout=dropout,
            lamda=lamda,
            alpha=alpha,
            variant=variant
        )

    def forward(self, g, adj, idx, lam=None, ws=None, mix_layer=None, aug=None):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if aug:
            input_ids1, attention_mask1 = g.ndata['input_ids1'][idx], g.ndata['attention_mask1'][idx]
            input_ids2, attention_mask2 = g.ndata['input_ids2'][idx], g.ndata['attention_mask2'][idx]
        else:
            input_ids1 = attention_mask1 = input_ids2 = attention_mask2 = None

        if self.training:
            cls_feats = self.xlnet_model(input_ids, input_ids1, input_ids2, lam, ws, mix_layer, attention_mask,
                                         attention_mask1, attention_mask2)[:, -1]
            # 把每个batch对应idx的特征(hidden)存储在图上对应idx位置上
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]

        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcnii_logit = self.gcnii(g.ndata['cls_feats'], adj)[idx]
        gcnii_pred = th.nn.Softmax(dim=1)(gcnii_logit)
        pred = (gcnii_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)

        return pred
