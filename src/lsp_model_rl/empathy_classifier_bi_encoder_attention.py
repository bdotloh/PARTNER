import torch
import codecs
import numpy as np


import pandas as pd
import re
import csv
import numpy as np
import sys

import time

from sklearn.metrics import f1_score

from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

# sys.path.insert(0, 'empathy_util/models.py')

from .empathy_util.models import BiEncoderAttentionWithRationaleClassification
from transformers import AdamW, RobertaConfig

import datetime



class EmpathyClassifier():

	def __init__(self, 
			device,
			ER_model_path = 'src/lsp_model_rl/empathy_util/empathy-models/ER-bi-encoder-attention.pth', 
			IP_model_path = 'src/lsp_model_rl/empathy_util/empathy-models/IP-bi-encoder-attention.pth',
			EX_model_path = 'src/lsp_model_rl/empathy_util/empathy-models/EX-bi-encoder-attention.pth',
			batch_size=2):
		
		self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
		self.batch_size = batch_size
		self.device = device

		self.model_ER = BiEncoderAttentionWithRationaleClassification()
		self.model_IP = BiEncoderAttentionWithRationaleClassification()
		self.model_EX = BiEncoderAttentionWithRationaleClassification()

		ER_weights = torch.load(ER_model_path, map_location=torch.device(device))
		self.model_ER.load_state_dict(ER_weights)

		IP_weights = torch.load(IP_model_path, map_location=torch.device(device))
		self.model_IP.load_state_dict(IP_weights)

		EX_weights = torch.load(EX_model_path, map_location=torch.device(device))
		self.model_EX.load_state_dict(EX_weights)

		self.model_ER.to(self.device)
		self.model_IP.to(self.device)
		self.model_EX.to(self.device)

		self.model_ER.eval()
		self.model_IP.eval()
		self.model_EX.eval()


	def predict_empathy(self, seeker_posts, response_posts):
		
		input_ids_SP = []
		attention_masks_SP = []
		
		for sent in seeker_posts:

			encoded_dict = self.tokenizer.encode_plus(
								sent,                      # Sentence to encode.
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = 64,           # Pad & truncate all sentences.
								truncation=True,
								pad_to_max_length = True,
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',     # Return pytorch tensors.
						)
			
			input_ids_SP.append(encoded_dict['input_ids'])
			attention_masks_SP.append(encoded_dict['attention_mask'])


		input_ids_RP = []
		attention_masks_RP = []

		for sent in response_posts:
			encoded_dict = self.tokenizer.encode_plus(
								sent,                      # Sentence to encode.
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = 64,           # Pad & truncate all sentences.
								truncation=True,
								padding = 'max_length',
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',     # Return pytorch tensors.
						)
			
			input_ids_RP.append(encoded_dict['input_ids'])
			attention_masks_RP.append(encoded_dict['attention_mask'])
		
		input_ids_SP = torch.cat(input_ids_SP, dim=0)
		attention_masks_SP = torch.cat(attention_masks_SP, dim=0)

		input_ids_RP = torch.cat(input_ids_RP, dim=0)
		attention_masks_RP = torch.cat(attention_masks_RP, dim=0)

		dataset = TensorDataset(input_ids_SP, attention_masks_SP, input_ids_RP, attention_masks_RP)

		dataloader = DataLoader(
			dataset, # The test samples.
			sampler = SequentialSampler(dataset), # Pull out batches sequentially.
			batch_size = self.batch_size # Evaluate with this batch size.
		)

		self.model_ER.eval()
		self.model_IP.eval()
		self.model_EX.eval()

		for batch in dataloader:
			b_input_ids_SP = batch[0].to(self.device)
			b_input_mask_SP = batch[1].to(self.device)
			b_input_ids_RP = batch[2].to(self.device)
			b_input_mask_RP = batch[3].to(self.device)

			with torch.no_grad():
				(logits_empathy_ER, logits_rationale_ER,) = self.model_ER(input_ids_SP = b_input_ids_SP,
														input_ids_RP = b_input_ids_RP, 
														token_type_ids_SP=None,
														token_type_ids_RP=None, 
														attention_mask_SP=b_input_mask_SP,
														attention_mask_RP=b_input_mask_RP)
				
				(logits_empathy_IP, logits_rationale_IP,) = self.model_IP(input_ids_SP = b_input_ids_SP,
														input_ids_RP = b_input_ids_RP, 
														token_type_ids_SP=None,
														token_type_ids_RP=None, 
														attention_mask_SP=b_input_mask_SP,
														attention_mask_RP=b_input_mask_RP)

				(logits_empathy_EX, logits_rationale_EX,) = self.model_EX(input_ids_SP = b_input_ids_SP,
														input_ids_RP = b_input_ids_RP, 
														token_type_ids_SP=None,
														token_type_ids_RP=None, 
														attention_mask_SP=b_input_mask_SP,
														attention_mask_RP=b_input_mask_RP)

				
			logits_empathy_ER = logits_empathy_ER.detach().cpu().numpy().tolist()
			predictions_ER = np.argmax(logits_empathy_ER, axis=1).flatten()

			logits_empathy_IP = logits_empathy_IP.detach().cpu().numpy().tolist()
			predictions_IP = np.argmax(logits_empathy_IP, axis=1).flatten()

			logits_empathy_EX = logits_empathy_EX.detach().cpu().numpy().tolist()
			predictions_EX = np.argmax(logits_empathy_EX, axis=1).flatten()

		return (logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX)


'''
Example:
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
seeker_posts = ['I need help', 'I want to talk to someone','I do not have any friends.']
response_posts = ['why do you feel this way?', 'I understand how you feel','do you want to talk about it?']

empathy_classifier = EmpathyClassifier(device)

(logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX) = empathy_classifier.predict_empathy(seeker_posts, response_posts)

print(predictions_ER, predictions_IP, predictions_EX)
