def config_data():
    config = {
        "batch_size" : 16,
        "max_length" : 16,
        "stride" : 8
    }
    return config

def config_model(tokenizer):
    config = {
        "vocab_size" : tokenizer.n_vocab,
        "emb_dim" : 256,
        "n_heads" : 4,
        "n_layers" : 12,
        "qkv_bias" : False,
        "context_length" : 32,
        "drop_rate" : 0.1,
        "max_length" : 16
    }
    return config