def linear_scheduler(model, weight_decay):
    named_parameters = list(model.named_parameters())
    optimizer_grouped_parameters = []
    no_decay = ['bias','LayerNorm.weight','LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    return optimizer_grouped_parameters


def layerwise_lrd(model, weight_decay,learning_rate, head_lr):
    named_parameters = list(model.named_parameters())
    optimizer_grouped_parameters = []
    no_decay = ['bias','LayerNorm.weight', 'LayerNorm.bias']
    init_lr = learning_rate 
    head_lr = head_lr
    lr = init_lr
    params_0 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n) 
                    and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if ("pooler" in n or "regressor" in n)
                    and not any(nd in n for nd in no_decay)]
        
    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}    
    optimizer_grouped_parameters.append(head_params)
            
    head_params = {"params": params_1, "lr": head_lr, "weight_decay": weight_decay}    
    optimizer_grouped_parameters.append(head_params)

    for layer in range(11,-1,-1):        
        params_0 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        
        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        optimizer_grouped_parameters.append(layer_params)   
                            
        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        optimizer_grouped_parameters.append(layer_params)       
        
        lr *= 0.9     

        # === Embeddings layer ==========================================================
        
    params_0 = [p for n,p in named_parameters if "embeddings" in n 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    
    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
    optimizer_grouped_parameters.append(embed_params)
        
    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay} 
    optimizer_grouped_parameters.append(embed_params)   

    return optimizer_grouped_parameters

def grouped_layerwise_lrd(model, weight_decay,learning_rate):
    
    named_parameters = list(model.named_parameters())
    optimizer_grouped_parameters = []
    no_decay = ['bias','LayerNorm.weight', 'LayerNorm.bias']
    set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
    set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]
    init_lr = learning_rate
    for i, (name, params) in enumerate(named_parameters):  
        
        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01

        if name.startswith("l1.embeddings") or name.startswith("l1.encoder"):            
            # For first set, set lr to 1e-6 (i.e. 0.000001)
            lr = init_lr       
            
            # For set_2, increase lr to 0.00000175
            lr = init_lr * 1.75 if any(p in name for p in set_2) else lr
            
            # For set_3, increase lr to 0.0000035 
            lr = init_lr * 3.5 if any(p in name for p in set_3) else lr
            
            optimizer_grouped_parameters.append({"params": params,
                                    "weight_decay": weight_decay,
                                    "lr": lr})  
            
        # For regressor and pooler, set lr to 0.0000036 (slightly higher than the top layer).                
        if name.startswith("regressor") or name.startswith("l1.pooler"):               
            lr = init_lr * 3.6 
            
            optimizer_grouped_parameters.append({"params": params,
                                    "weight_decay": weight_decay,
                                    "lr": lr})    
    
    return optimizer_grouped_parameters