def config_data():
    config = {
        "batch_size" : 16,
        "max_length" : 4,
        "stride" : 4
    }
    return config

def config_model(tokenizer):
    config = {
        "vocab_size" : tokenizer.n_vocab(),
        "emb_dim" : 768,
        "n_heads" : 12,
        "n_layers" : 12,
        "qkv_bias" : False,
        "context_length" : 32,
        "drop_rate" : 0.0,
        "max_length" : 4
    }
    
    return config