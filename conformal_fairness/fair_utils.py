import torch


def inverse_quantile(x, target):
    n = x.shape[0]
    sorted_x = torch.sort(x)

    # Get the index of the largest value less than or equal to the target.
    # Add 1 to make it 1-indexed, instead of 0-indexed
    satisfied = torch.where(sorted_x.values <= target)[0].reshape(-1)
    if len(satisfied) > 0:
        return 1 - (satisfied[-1] + 1) / (n + 1)
    return 1
