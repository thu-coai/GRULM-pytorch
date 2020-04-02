# coding:utf-8
import logging

import torch
from torch import nn

from utils import zeros, LongTensor,\
			BaseNetwork, MyGRU, Storage, gumbel_max, flattenSequence, SingleGRU, SequenceBatchNorm

# pylint: disable=W0221
class Network(BaseNetwork):
	def __init__(self, param):
		super().__init__(param)

		self.embLayer = EmbeddingLayer(param)
		self.genNetwork = GenNetwork(param)

	def forward(self, incoming):
		incoming.result = Storage()

		self.embLayer.forward(incoming)
		self.genNetwork.forward(incoming)

		incoming.result.loss = incoming.result.word_loss

		if torch.isnan(incoming.result.loss).detach().cpu().numpy() > 0:
			logging.info("Nan detected")
			logging.info(incoming.result)
			raise FloatingPointError("Nan detected")

	def detail_forward(self, incoming):
		incoming.result = Storage()

		self.embLayer.detail_forward(incoming)
		self.genNetwork.detail_forward(incoming)

class EmbeddingLayer(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param
		volatile = param.volatile

		self.embLayer = nn.Embedding(volatile.dm.frequent_vocab_size, args.embedding_size)
		self.embLayer.weight = nn.Parameter(torch.Tensor(volatile.wordvec))

	def forward(self, incoming):
		incoming.sent = Storage()
		incoming.sent.embedding = self.embLayer(incoming.data.sent)
		incoming.sent.embLayer = self.embLayer

	def detail_forward(self, incoming):
		incoming.sent = Storage()
		incoming.sent.embLayer = self.embLayer

class GenNetwork(nn.Module):
	def __init__(self, param):
		super().__init__()
		self.args = args = param.args
		self.param = param

		self.GRULayer = SingleGRU(args.embedding_size, args.dh_size, initpara=True)
		self.wLinearLayer = nn.Linear(args.dh_size, param.volatile.dm.frequent_vocab_size)
		self.lossCE = nn.CrossEntropyLoss()
		self.start_generate_id = param.volatile.dm.go_id

		self.drop = nn.Dropout(args.droprate)

	def teacherForcing(self, inp, gen):
		embedding = inp.embedding
		length = inp.length
		embedding = self.drop(embedding)
		_, gen.h = self.GRULayer.forward(embedding, length-1)
		gen.h = torch.stack(gen.h, dim=0)
		gen.h = self.drop(gen.h)
		gen.w = self.wLinearLayer(gen.h)

	def freerun(self, inp, gen):
		#mode: beam = beamsearch; max = choose max; sample = random_sampling; sample10 = sample from max 10

		def wLinearLayerCallback(gru_h):
			gru_h = self.drop(gru_h)
			w = self.wLinearLayer(gru_h) / self.args.temperature
			return w

		def input_callback(i, now):
			return self.drop(now)

		if self.args.decode_mode == "beam":
			new_gen = self.GRULayer.beamsearch(inp, self.args.top_k, wLinearLayerCallback, \
				input_callback=input_callback, no_unk=True, length_penalty=self.args.length_penalty)
			w_o = []
			length = []
			for i in range(inp.batch_size):
				w_o.append(new_gen.w_o[:, i, 0])
				length.append(new_gen.length[i][0])
			gen.w_o = torch.stack(w_o).transpose(0, 1)
			gen.length = length

		else:
			new_gen = self.GRULayer.freerun(inp, wLinearLayerCallback, self.args.decode_mode, \
				input_callback=input_callback, no_unk=True, top_k=self.args.top_k)
			gen.w_o = new_gen.w_o
			gen.length = new_gen.length

	def forward(self, incoming):
		inp = Storage()
		inp.length = incoming.data.sent_length
		inp.embedding = incoming.sent.embedding

		incoming.gen = gen = Storage()
		self.teacherForcing(inp, gen)

		w_o_f = flattenSequence(gen.w, incoming.data.sent_length-1)
		data_f = flattenSequence(incoming.data.sent[1:], incoming.data.sent_length-1)
		incoming.result.word_loss = self.lossCE(w_o_f, data_f)
		incoming.result.perplexity = torch.exp(incoming.result.word_loss)

	def detail_forward(self, incoming):
		inp = Storage()
		batch_size = inp.batch_size = incoming.data.batch_size
		inp.embLayer = incoming.sent.embLayer
		inp.dm = self.param.volatile.dm
		inp.max_sent_length = self.args.max_sent_length

		incoming.gen = gen = Storage()
		self.freerun(inp, gen)

		dm = self.param.volatile.dm
		w_o = gen.w_o.detach().cpu().numpy()
		incoming.result.sent_str = sent_str = \
				[" ".join(dm.convert_ids_to_tokens(w_o[:, i].tolist())) for i in range(batch_size)]
		incoming.result.golden_str = golden_str = \
				[" ".join(dm.convert_ids_to_tokens(incoming.data.sent[:, i].detach().cpu().numpy().tolist()))\
				for i in range(batch_size)]
		incoming.result.show_str = "\n".join(["sent: " + a + "\n" + \
				"golden: " + b + "\n" \
				for a, b in zip(sent_str, golden_str)])
