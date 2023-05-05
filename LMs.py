import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

try:
    from transformers import BertTokenizer, BertModel
    from transformers import RobertaTokenizer, RobertaModel
except ImportError:
    print(
        "Please install transformer from HuggingFace to use the BERT language models!"
    )


class HuggingFaceBERT(nn.Module):
    def __init__(
        self,
        source,
        save_path,
        device="cpu",
        pooled=False,
        output_norm=False,
        freeze=True,
        pretrain=True,
    ):
        super().__init__()
        self.device = device
        self.pooled = pooled
        
        # Download the model from HuggingFace.
        # if pretrain is False, we do not download the pretrained weights
        # it it is True, we download and load them.
        # if not (pretrain):
        #     config = config.from_pretrained(source, cache_dir=save_path)
        #     self.model = model(config)
        # else:
        #     self.model = BertModel.from_pretrained(source, cache_dir=save_path)

        if "bert-base" in source:
            self.tokenizer = BertTokenizer.from_pretrained(source, cache_dir=save_path)
            self.model = BertModel.from_pretrained(source, cache_dir=save_path)

        if "roberta-base" in source:
            self.tokenizer = RobertaTokenizer.from_pretrained(source, cache_dir=save_path)
            self.model = RobertaModel.from_pretrained(source, cache_dir=save_path)

        if "roberta-large" in source:
            self.tokenizer = RobertaTokenizer.from_pretrained(source, cache_dir=save_path)
            self.model = RobertaModel.from_pretrained(source, cache_dir=save_path)

        self.freeze = freeze
        self.output_norm = output_norm
        if self.freeze:
            self.model.eval()
        else:
            self.model.train()

    def forward(self, txt):
        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(txt).detach()

        return self.extract_features(txt)

    def extract_features(self, txts):
        outs = []
        idx = 1 if self.pooled else 0
        for txt in txts:
            encoded_txt = self.tokenizer(txt, return_tensors='pt').to(self.device)
            out = self.model(**encoded_txt)[idx]
            if not self.pooled: out = out.squeeze(0)
            outs.append(out)
        # out = torch.cat(outs, 0)
        out = pad_sequence(outs, batch_first=True, padding_value=0.0)
        # print("LM some weights", self.model.encoder.layer[0].output.dense.weight)
        # print("LM extract_features out", out.size())

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out