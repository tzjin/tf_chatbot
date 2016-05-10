import pickle
import util.tokenizer
import re
import os
import tensorflow as tf
from tensorflow.python.platform import gfile

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

word_split = re.compile(b"([.,!?\"':;)(])")
digits = re.compile(br"\d")

class VocabBuilder(object):


	# builds vocab file
	def __init__(self, max_voc, d_path, tokenzr=None, norm_dig=True):
		if tokenzr is None:
			self.tokenzr = util.tokenizer.basic_tokenizer
		else:
			self.tokenzr = tokenzr

		self.d_path = d_path
		self.max_voc = max_voc
		self.vocab = {}

	def growVocab(self, text, norm_dig=True):
		tokenizer = self.tokenzr

		word_toks = tokenizer(text)

		for tok in word_toks:
			word = tok
			if norm_dig:
				word = re.sub(digits, b"0", tok)

			if word in self.vocab:
				self.vocab[word] += 1
			else:
				self.vocab[word] = 1

	def createVocabFile(self):
		voc_ls = _START_VOCAB + sorted(self.vocab, key=self.vocab.get, reverse=True)
		last_back = min(self.max_voc, len(voc_ls))
		voc_ls = voc_ls[:last_back]

		voc_path = os.path.join(self.d_path, "vocab.txt")

		with gfile.GFile(voc_path, mode="wb") as file:
			for voc_word in voc_ls:
				file.write(voc_word + b"\n")


class VocabMapper(object):

	def __init__(self, d_path, tokenzr=None):
		if tokenzr is None:
			self.tokenzr = util.tokenizer.basic_tokenizer
		else:
			self.tokenzr = tokenzr

		voc_path = os.path.join(self.d_path, "vocab.txt")
		exists = gfile.Exists(voc_path)

		if exists:
			rev_voc = []
			with gfile.GFile(voc_path, mode="rb") as file:
				rev_voc.extend(file.readlines())
			rev_voc = [line.strip() for line in rev_voc]
			enumerated = enumerate(rev_voc)
			reg_voc = dict([(first, second) for (second, first) in enumerated])
			self.vocab = reg_voc
			self.rev_voc = rev_voc

		else:
			raise ValueError("File not found.")

	def getVocabSize(self):
		return len(self.rev_voc)

	def tokens2Indices(self, text):
		if type(text) == type("string"):
			text = self.tokenzr(text)

		ret_indices = []
		for tok in text:
			if tok in self.vocab:
				index = self.vocab[tok]
				ret_indices.append(index)
			else:
				ret_indices.append(UNK_ID)
		return ret_indices


	def indices2Tokens(self, indices):
		toks = []
		for i in indices:
			tok = self.rev_voc[i]
			toks.append(tok)
		return toks
