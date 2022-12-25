from flask import Flask,request,jsonify,render_template

import json
import numpy as np
import torch
import require

from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 

class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv('wellness.csv')
        self.train_set = CharDataset(data, max_len=self.hparams.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader
    
    

    

    
    def chat(self, q="",sent='0'):
        tok = TOKENIZER
        sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            
            a=""
            if q == 'quit':
                return
            else:
                real_result=""
                
                send=""
                input_ids=torch.LongTensor(tok.encode(U_TKN+q+SENT+sent+S_TKN+a)).unsqueeze(dim=0)
                pred=self(input_ids)
                
                info=require.check_similarity(q)
                
                real_result+="Cosine similarity: "+str(info['similarity'])+"\n"
                real_result+="검색 기반: "+info['answer']+"\n"
                #kogpt로 문장생성해야하는 경우
                
                while(1):    
                    input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                    pred = self(input_ids)
                    proba,result=torch.topk(pred,k=3,dim=-1)
                    result_array=result.numpy()
                    print("prob:",proba)
                    print("result:",result)
                    send+="prob: "+str(proba[-1][-1])
                    result_array=result_array.swapaxes(1,2)
                    gen=tok.convert_ids_to_tokens(
                        result_array[0][0].tolist()
                            )[-1]
                    gen1=tok.convert_ids_to_tokens(
                        result_array[0][1].tolist()
                            )[-1]
                            
                    gen2=tok.convert_ids_to_tokens(
                            result_array[0][2].tolist()
                            )[-1]
                    
                    
                    if gen==EOS:
                        break
                    a+=gen.replace('▁',' ')

                real_result+="kogpt로 문장생성: "+a+"\n"
                real_result+="*"
                if info['kogpt']:
                    
                    
                    real_result+="(G) "+a

                #검색기반챗봇
                else:
                    
                    a=require.get_string(q)
                    real_result+="(S) "+a
                
                
                    


        return real_result

    def unity_chat(self, q="",sent='0'):
        tok = TOKENIZER
        sent_tokens = tok.tokenize(sent)
        real_result=""
        with torch.no_grad():
            
            a=""
            if q == 'quit':
                return
            else:
                
                input_ids=torch.LongTensor(tok.encode(U_TKN+q+SENT+sent+S_TKN+a)).unsqueeze(dim=0)
                pred=self(input_ids)
                
                info=require.check_similarity(q)
                
                
                
                while(1):    
                    input_ids = torch.LongTensor(tok.encode(U_TKN + q + SENT + sent + S_TKN + a)).unsqueeze(dim=0)
                    pred = self(input_ids)
                    proba,result=torch.topk(pred,k=3,dim=-1)
                    result_array=result.numpy()
                    print("prob:",proba)
                    print("result:",result)
                    result_array=result_array.swapaxes(1,2)
                    gen=tok.convert_ids_to_tokens(
                        result_array[0][0].tolist()
                            )[-1]
                    gen1=tok.convert_ids_to_tokens(
                        result_array[0][1].tolist()
                            )[-1]
                            
                    gen2=tok.convert_ids_to_tokens(
                            result_array[0][2].tolist()
                            )[-1]
                    
                    if gen==EOS:
                        break
                    a+=gen.replace('▁',' ')

                
                if info['kogpt']:
                    
                    
                    real_result="(G) "+a

                #검색기반챗봇
                else:
                    
                    a=require.get_string(q)
                    real_result="(S) "+a
                
                
                    


        return real_result




model=KoGPT2Chat.load_from_checkpoint("model_chp/model_-last.ckpt")
app=Flask(__name__)


@app.route("/",methods=['GET','POST'])
def hello():
    
    return render_template("index.html")

@app.route("/answer",methods=['POST'])
def get_answer():
    
    content=request.get_json(silent=True) #질문받기
    answer=model.chat(content["question"]) #답변처리
    
    return answer

@app.route("/unity_answer",methods=['POST'])
def send():
    content=request.get_json(silent=True)
    answer=model.unity_chat(content["question"])

    return answer


if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)