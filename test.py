import torch
from transformers import *

'''
MODEL = (RobertaModel, RobertaTokenizer, 'roberta-base')

model_class, tokenizer_class, pretrained_weights = MODEL
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

input_ids = torch.tensor([tokenizer.encode("Here is some text to encode",
                                           add_special_tokens=True)])
with torch.no_grad():
    last_hidden_states=model(input_ids)[0]
'''
BERT_QA_MODEL = BertForQuestionAnswering

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

# model=BERT_QA_MODEL.from_pretrained(pretrained_weights)
model = BERT_QA_MODEL.from_pretrained(pretrained_weights,
                                      output_hidden_states=True,
                                      output_attentions=True)
input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])

all_hiddden_states, all_attentions = model(input_ids)[-2:]
print(all_hiddden_states)
