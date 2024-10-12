def config_data():
    config = {
        "batch_size" : 12,
        "max_length" : 64,
        "stride" : 1
    }
    return config

def config_model(tokenizer):
    config = {
        "vocab_size" : tokenizer.n_vocab,
        "emb_dim" : 128,
        "n_heads" : 4,
        "n_layers" : 4,
        "qkv_bias" : False,
        "context_length" : 64,
        "drop_rate" : 0.0,
        "max_length" : 64
    }
    
    return config