# -*- coding: utf-8 -*-

#####################################################################
# File Name:  CBOW.py
# Author: shenming
# Created Time: Mon Jan 13 00:01:27 2020
#####################################################################

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        sum_embeds = torch.sum(embeds, dim = 0).view((1, -1))
        out = self.linear1(sum_embeds);
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs;

    def embs(self, ids):
        return self.embeddings(ids)

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)


make_context_vector(data[0][0], word_to_ix)  # example



if __name__ == "__main__":
    loss_function = nn.NLLLoss()
    model = CBOW(len(vocab), 128, CONTEXT_SIZE * 2)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(100):
        total_loss = 0
        for context, target in data:
            context_idx = make_context_vector(context, word_to_ix)

            model.zero_grad()

            log_probs = model(context_idx)

            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype = torch.long))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)
    print(losses)
    print(model.embs(torch.tensor([word_to_ix['we'], word_to_ix['they']], dtype = torch.long)))
