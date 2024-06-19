import torch

def get_distribution_stats(X):
    mean = torch.mean(X).item()
    var = torch.var(X).item()
    median = torch.median(X).item()

    return {"mean": mean, "var": var, "median": median}

def get_tensor_metrics(X):
    def get_average_l1_norm(x):
        return torch.flatten(torch.linalg.vector_norm(x, ord=1, dim=(2, 3)))

    def get_average_l2_norm(x):
        return torch.flatten(torch.linalg.vector_norm(x, ord=2, dim=(2, 3)))

    l1s = get_average_l1_norm(X)
    l2s = get_average_l2_norm(X)
    code_sparsities = l1s / l2s

    return {
        "l1": torch.mean(l1s).item(),
        "l2": torch.mean(l2s).item(),
        "code_sparsity": torch.mean(code_sparsities).item(),
    }

def get_matrix_metrics(X):
    if torch.isnan(X).any():
        return
    if torch.isinf(X).any():
        return
    if torch.isneginf(X).any():
        return

    def get_flattened_l1_norm(x):
        return torch.linalg.vector_norm(x, ord=1)

    def get_flattened_l2_norm(x):
        return torch.linalg.vector_norm(x, ord=2)

    def get_spectral_norm(X):
        return torch.linalg.matrix_norm(X, ord=2)

    l1 = get_flattened_l1_norm(X).item()
    l2 = get_flattened_l2_norm(X).item()

    trace = torch.trace(X).item()
    spectral = get_spectral_norm(X).item()
    singular_vals = torch.svd(X, compute_uv=False).S
    singular_vals[singular_vals < 1e-5] = 0.0
    mean = torch.mean(singular_vals).item()
    var = torch.var(singular_vals).item()

    return {
        "l1": l1,
        "l2": l2,
        "trace": trace,
        "spectral": spectral,
        "code_sparsity": l1 / l2,
        "computational_sparsity": trace / spectral,
        "mean_singular_value": mean,
        "var_singular_value": var,
        "singular_values": singular_vals.tolist(),
    }

@torch.no_grad()
def get_metrics_resnet18(model):
    data_dict = {
        "w": [],
    }
    weights = []
    biases = []

    data_dict["w"].append(get_tensor_metrics(model.conv1.weight))
    weights.append(model.conv1.weight.flatten())

    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            data_dict["w"].append(get_tensor_metrics(block.conv1.weight))
            data_dict["w"].append(get_tensor_metrics(block.conv2.weight))
            weights.append(block.conv1.weight.flatten())
            weights.append(block.conv2.weight.flatten())

    data_dict["w"].append(get_matrix_metrics(model.linear.weight))
    weights.append(model.linear.weight.flatten())
    biases.append(model.linear.bias.flatten())

    data_dict["w_all"] = get_distribution_stats(torch.cat(weights))
    data_dict["b_all"] = get_distribution_stats(torch.cat(biases))

    return data_dict