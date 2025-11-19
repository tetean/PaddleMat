def count_parameters(model):
    """计算模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if not p.stop_gradient)

    return total_params, trainable_params


def print_model_info(model):
    """打印模型信息，包括参数量"""
    total_params, trainable_params = count_parameters(model)

    print(f"\nModel Information:")
    print(f"=" * 50)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print(f"Model size (approx):  {total_params * 4 / 1024 / 1024:.2f} MB")  # 假设float32
    print(f"=" * 50)