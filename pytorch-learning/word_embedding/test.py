# -*- coding: utf-8 -*-

#####################################################################
# File Name:  test.py
# Author: shenming
# Created Time: Sun Jan 12 23:16:57 2020
#####################################################################

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if __name__ == "__main__":
    torch.manual_seed(1)

    word_to_ix = {"hello": 0, "world": 1}
    embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
    lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
    hello_embed = embeds(lookup_tensor)
    print(hello_embed)
