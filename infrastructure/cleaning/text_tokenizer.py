import jieba

def jieba_tokenizer(text):
    return [tok for tok in jieba.lcut(text) if tok.strip()]