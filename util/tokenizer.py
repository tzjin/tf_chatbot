import re

word_split = re.complie(b"([.,!?\"':;)(])")


def basic_tokenizer(seq):
	words = []
	sent = seq.strip().split()
	for word in sent:
		words.extend(re.split(word_split, word))

	ls = [word for word in words if word]

	return ls

def character_tokenizer(seq):
	return list(seq)
