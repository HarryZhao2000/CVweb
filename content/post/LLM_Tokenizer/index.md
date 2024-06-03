---
title: LLM Tokenizer from scratch
subtitle: BPE, WorPiece, and Unigram tokenizers
date: 2024-05-28T00:00:00.000Z
summary: Here's a brief overviwe of common LLM tokenizers, especially subword tokenizer.
draft: false
featured: false
authors:
  - Haoran Zhao
lastmod: 2024-06-03T00:00:00.000Z
tags:
  - LLM
  - NLP
categories: []
projects:
  - LLM
---
## What is a tokenizer?
The process of breaking down text into **smaller subword units**, known as **tokens**.
It is the **first** step and the **last** step of text processing and modeling

## Type of tokenizer
- Word Tokenization: Break text into words. Most common and effective for languages with clear boundaries, like English
	- May have very large vocabulary
	- High frequency of unknown words
	- Same words may in different tokens, like `cat` and `cats`
- Character Tokenization: Break text into characters. More granular and can be especially useful for certain languages or specific NLP tasks.
	- Token sequence may be very long
	- Less information in one single token
- Subword Tokenization: Break text into subwords. It's useful for languages with complex morphology, such as German or Finnish.
	- Common words should not be break into subwords
	- Less common words should be represented by common subwords

## Common Subword Tokenizers
### Byte Pair Encoding (BPE)

1. Firstly, we can define need to do the pre-tokenization, and have a token list. We assume that we have a list of integers in range 0..255 representing the tokens from a given text for convenience
2. Then, we need to set the desired size of vocabulary as the hyper parameter. Here we use 267 as example
```
vocab_size = 267
num_merges = vocab_size - 256 # We need to merge vocab_size - 256 times
tokens_list = list(tokens)
```
1. Then, we need to calculate the frequency of pairs of consecutive tokens for `num_merges` rounds, and merge the most frequent pair of each round as a new token. 
	1. Firstly we calculate the frequency, and find the most frequent pair of each round
```
def get_pair(tokens):
    counts = {}
    for pair in zip(tokens, tokens[1:]): # pairs of consecutive tokens
        counts[pair] = counts.get(pair, 0) + 1
    max_pair = max(counts, key=counts.get)
    return max_pair, counts[max_pair]
```
	2. Then, we merge this pair as a new token
```
def merge(tokens_list, pair, new_token_id):
    new_tokens_list = []
    i = 0
    while i < len(tokens_list):
        if i < len(tokens_list) - 1 and tokens_list[i] == pair[0] and tokens_list[i+1] == pair[1]: # if found the pair
            new_tokens_list.append(new_token_id)
            i += 2 # skip the original tokens
        else:
            new_tokens_list.append(tokens_list[i])
            i += 1
    return new_tokens_list
```
2. Finally, we can get the new_tokens_list via BPE
```
merges = {} # mapping from old token pair to new token
for i in range(num_merges):
pair, _ = get_pair(tokens_list)
new_token_id = 256 + i
print(f"merging {pair} into a new token {new_token_id}")
tokens_list = merge(tokens_list, pair, new_token_id)
merges[pair] = new_token_id
```
3.  Thus, we now have a simple BPE tokenizer:
```
class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        self.merges = {}

    def encode(self, sentence):
        tokens = list(map(int, sentence.encode("utf-8")))
        return tokens

    def get_pair(self, tokens):
        counts = {}
        for pair in zip(tokens, tokens[1:]):  # pairs of consecutive tokens
            counts[pair] = counts.get(pair, 0) + 1
        max_pair = max(counts, key=counts.get)
        return max_pair, counts[max_pair]

    def merge(self, tokens_list, pair, new_token_id):
        new_tokens_list = []
        i = 0
        while i < len(tokens_list):
            if i < len(tokens_list) - 1 and tokens_list[i] == pair[0] and tokens_list[i + 1] == pair[1]:  # if found the pair
                new_tokens_list.append(new_token_id)
                i += 2  # skip the original tokens
            else:
                new_tokens_list.append(tokens_list[i])
                i += 1
        return new_tokens_list

    def train(self, sentence):
        tokens_list = self.encode(sentence)
        for i in range(self.num_merges):
            pair, _ = self.get_pair(tokens_list)
            new_token_id = 256 + i
            print(f"merging {pair} into a new token {new_token_id}")
            tokens_list = self.merge(tokens_list, pair, new_token_id)
            self.merges[pair] = new_token_id

    def tokenize(self, sentence):
        tokens = self.encode(sentence)
        while True:
            pairs = [pair for pair in zip(tokens, tokens[1:]) if pair in self.merges]
            if not pairs:
                break
            for pair in pairs:
                new_token_id = self.merges[pair]
                tokens = self.merge(tokens, pair, new_token_id)
        return tokens

example_sentence = "In Hugging Face Transformers, when a tokenizer uses BPE, it means that it is using Byte Pair Encoding to tokenize and represent words or subwords. This approach is popular in transformer-based models because it allows them to handle a vast vocabulary efficiently and capture both common and rare words effectively."

tokenizer = BPETokenizer(vocab_size=276)
tokenizer.train(example_sentence)
encoded_tokens = tokenizer.tokenize(example_sentence)
print("Encoded Tokens:", encoded_tokens)

```
### WordPiece
Similar to BPE, however:
- Using `##` as prefix except the first subword.
- Calculating frequency of each subword, rather than pairs
- Using frequency to calculate pair score:

$$
Pair Score = \frac{\text{Frequency of pair}}{\text{Frequency of token 1} \times \text{Frequency of Token 2}}
$$

- Higher score means better matching between tokens, merge pair with highest score for `num_merges` rounds

1. Firstly, we need to add `##` prefix to subwords except the first one of each tokens
```
def encode(sentence, vocab):
	tokens = sentence.split()
	encoded_tokens = []
	for token in tokens:
		sub_tokens = list(token)
		for i, sub_token in enumerate(sub_tokens):
			if i == 0:
				if sub_token not in self.vocab:
					vocab[sub_token] = len(self.vocab)
				encoded_tokens.append(sub_token)
			else:
				prefixed_token = "##" + sub_token
				if prefixed_token not in self.vocab:
					vocab[prefixed_token] = len(self.vocab)
				encoded_tokens.append(prefixed_token)
	return encoded_tokens
```

1. Then, we need to calculate the Pair Score:
```
def get_pair_scores(self, tokens): 
	pair_freq = {} 
	for i in range(len(tokens) - 1): 
		pair = (tokens[i], tokens[i + 1]) 
		if pair in pair_freq: 
			pair_freq[pair] += 1 
		else: 
			pair_freq[pair] = 1 
	pair_scores = {} 
	for pair, freq in pair_freq.items(): 
		pair_scores[pair] = freq / (self.vocab[pair[0]] + self.vocab[pair[1]]) 
	return pair_scores
```
1. Then, we merge this pair as a new token just like BPE
2. During training, we need to keep eyes on handling the `##` prefix
```
tokens_list = encode(sentence)
num_merges = vocab_size - len(set(tokens_list)
for i in range(num_merges):
	pair_scores = get_pair_scores(tokens_list)
	if not pair_scores:
		break
	best_pair = max(pair_scores, key=pair_scores.get)
	new_token = best_pair[0] + best_pair[1].replace("##", "")
	if "##" in best_pair[1] and not best_pair[0].startswith("##"):
		new_token = best_pair[0] + best_pair[1].replace("##", "")
		new_token = "##" + new_token
	elif best_pair[0].startswith("##"):
		new_token = best_pair[0] + best_pair[1].replace("##", "")
	else:
		new_token = best_pair[0] + best_pair[1]
	print(f"Merging {best_pair} into new token '{new_token}'")
	tokens_list = self.merge_tokens(tokens_list, best_pair, new_token)
	vocab[new_token] = len(self.vocab)
	merges[best_pair] = new_token
```
1. Finally, we have a simple WordPiece Tokenizer:
```
class WordPieceTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}

    def encode(self, sentence):
        tokens = sentence.split()
        encoded_tokens = []
        for token in tokens:
            sub_tokens = list(token)
            for i, sub_token in enumerate(sub_tokens):
                if i == 0:
                    if sub_token not in self.vocab:
                        self.vocab[sub_token] = len(self.vocab)
                    encoded_tokens.append(sub_token)
                else:
                    prefixed_token = "##" + sub_token
                    if prefixed_token not in self.vocab:
                        self.vocab[prefixed_token] = len(self.vocab)
                    encoded_tokens.append(prefixed_token)
        return encoded_tokens

    def get_pair_scores(self, tokens):
        pair_freq = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in pair_freq:
                pair_freq[pair] += 1
            else:
                pair_freq[pair] = 1

        pair_scores = {}
        for pair, freq in pair_freq.items():
            pair_scores[pair] = freq / (self.vocab[pair[0]] + self.vocab[pair[1]])
        return pair_scores

    def merge_tokens(self, tokens_list, pair, new_token):
        new_tokens_list = []
        i = 0
        while i < len(tokens_list):
            if i < len(tokens_list) - 1 and tokens_list[i] == pair[0] and tokens_list[i + 1] == pair[1]:
                new_tokens_list.append(new_token)
                i += 2  # skip the original tokens
            else:
                new_tokens_list.append(tokens_list[i])
                i += 1
        return new_tokens_list

    def train(self, sentence):
        tokens_list = self.encode(sentence)
        num_merges = self.vocab_size - len(set(tokens_list))
        for i in range(num_merges):
            pair_scores = self.get_pair_scores(tokens_list)
            if not pair_scores:
                break
            best_pair = max(pair_scores, key=pair_scores.get)
            new_token = best_pair[0] + best_pair[1].replace("##", "")
            if "##" in best_pair[1] and not best_pair[0].startswith("##"):
                new_token = best_pair[0] + best_pair[1].replace("##", "")
                new_token = "##" + new_token
            elif best_pair[0].startswith("##"):
                new_token = best_pair[0] + best_pair[1].replace("##", "")
            else:
                new_token = best_pair[0] + best_pair[1]
            print(f"Merging {best_pair} into new token '{new_token}'")
            tokens_list = self.merge_tokens(tokens_list, best_pair, new_token)
            self.vocab[new_token] = len(self.vocab)
            self.merges[best_pair] = new_token

    def tokenize(self, sentence):
        tokens = self.encode(sentence)
        while True:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            merge_candidates = [pair for pair in pairs if pair in self.merges]
            if not merge_candidates:
                break
            for pair in merge_candidates:
                new_token = self.merges[pair]
                tokens = self.merge_tokens(tokens, pair, new_token)
        return tokens

example_sentence = "In Hugging Face Transformers, when a tokenizer uses WordPiece, it means that it is using WordPiece Encoding to tokenize and represent words or subwords. This approach is popular in transformer-based models because it allows them to handle a vast vocabulary efficiently and capture both common and rare words effectively."

tokenizer = WordPieceTokenizer(vocab_size=50)
tokenizer.train(example_sentence)
encoded_tokens = tokenizer.tokenize(example_sentence)
print("Encoded Tokens:", encoded_tokens)
```

## Unigram
Compared with BPE and WordPiece: Unigram iteratively removes **low-frequency** subwords to **reduce** the vocabulary size, rather than merging high-frequency pairs

1. Firstly, we need to create the initial vocab, generate all possible subwords
```
def initial_vocab(sentences): 
	token_counts = {} 
	for sentence in sentences: 
		tokens = list(sentence) 
		for token in tokens: 
			if token in token_counts: 
				token_counts[token] += 1 
			else: 
				token_counts[token] = 1 
	return token_counts
```
2. Then, we will calculate probability of each token
```
def calculate_probabilities(token_counts):
	total_count = sum(token_counts.values())
	token_probs = {token: count / total_count for token, count in token_counts.items()}
	return token_probs
```
3. Then, we will prune vocabulary based on the probability of each token, i.e. remove low-probability subwords
```
def prune_vocab(token_probs):
	# Sort tokens by probability
	sorted_tokens = sorted(token_probs.items(), key=lambda item: item[1])
	# Remove the token with the lowest probability
	token_to_remove = sorted_tokens[0][0]
	del token_probs[token_to_remove]
```
4. Then, we will recalculate the remaining subword probabilities, and iterate pruning and recalculation until reach the desired `vocab_size`
5. Finally, we now have a simple Unigram Tokenizer.

```
class UnigramTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.token_counts = {}

    def encode(self, sentence):
        return list(sentence)  # Unigram tokenizer splits into individual characters

    def initial_vocab(self, sentences):
        token_counts = {}
        for sentence in sentences:
            tokens = self.encode(sentence)
            for token in tokens:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1
        self.token_counts = token_counts

    def calculate_probabilities(self):
        total_count = sum(self.token_counts.values())
        token_probs = {token: count / total_count for token, count in self.token_counts.items()}
        return token_probs

    def prune_vocab(self, token_probs):
        # Sort tokens by probability
        sorted_tokens = sorted(token_probs.items(), key=lambda item: item[1])
        # Remove the token with the lowest probability
        token_to_remove = sorted_tokens[0][0]
        del self.token_counts[token_to_remove]

    def train(self, sentences):
        self.initial_vocab(sentences)
        
        while len(self.token_counts) > self.vocab_size:
            token_probs = self.calculate_probabilities()
            self.prune_vocab(token_probs)

        self.vocab = {token: idx for idx, token in enumerate(self.token_counts)}

    def tokenize(self, sentence):
        tokens = self.encode(sentence)
        return [self.vocab[token] for token in tokens if token in self.vocab]

# Example usage
example_sentences = [
    "In Hugging Face Transformers, when a tokenizer uses Unigram, it means that it is using individual characters or subwords as tokens.",
    "This approach is simpler than BPE and can be useful in certain contexts."
]

tokenizer = UnigramTokenizer(vocab_size=50)
tokenizer.train(example_sentences)
encoded_tokens = tokenizer.tokenize("In Hugging Face Transformers")
print("Encoded Tokens:", encoded_tokens)
```

## Pros and Cons
[Source](https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46)
### BPE
**Pros**： 
- Effectively handles out-of-vocabulary (OOV) words by breaking them down into smaller subwords.
- Captures common word and subword combinations, providing a good balance.
**Cons:**
- The training process is relatively complex, requiring multiple scans of the entire corpus.
- The merge rules are static and do not adapt to new data.
### WordPiece
**Pros**： 
- Better handles multilingual text and low-frequency words, especially for OOV words.
- Captures semantically relevant subword units.
**Cons:**
- The training process is complex, requiring probability distribution calculations.
- Like BPE, the merge rules are static.
### Unigram
**Pros**： 
- The training process allows dynamic adjustment of the vocabulary, providing more flexibility.
- Performs well in handling low-frequency words.
- Captures subword probability distribution more accurately.
**Cons:**
- The training process is complex, requiring multiple iterations and probability calculations.
- The initial vocabulary can be very large, leading to high computational cost.