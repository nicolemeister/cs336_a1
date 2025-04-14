
import torch
import math



def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    '''
    Write a function to apply the softmax operation on a tensor. Your function should
    take two parameters: a tensor and a dimension i, and apply softmax to the i-th dimension of the input
    tensor. The output tensor should have the same shape as the input tensor, but its i-th dimension will
    now have a normalized probability distribution. Use the trick of subtracting the maximum value in
    the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues.

    Args:
        tensor: torch.Tensor Input tensor to apply softmax to
        dim: int Dimension to apply softmax to (i) 

    Returns:
        torch.Tensor: Tensor with softmax applied to the specified dimension
    '''
    normalized_x = x - torch.max(x, dim=dim, keepdim=True)[0]
    return torch.exp(normalized_x) / torch.sum(torch.exp(normalized_x), dim=dim, keepdim=True)



def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    '''
    Implement the scaled dot-product attention function. Your implementation should
    handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
    (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
    dimensions (if provided). The implementation should return an output with the shape (batch_size,
    ..., d_v). See section 3.3 for a discussion on batch-like dimensions.
    Your implementation should also support an optional user-provided boolean mask of shape (seq_len,
    seq_len). The attention probabilities of positions with a mask value of True should collectively sum
    to 1, and the attention probabilities of positions with a mask value of False should be zero.
    Args: 
        Q: (batch_size, ..., seq_len, d_k) s,k
        K: (batch_size, ..., seq_len, d_k) s,k
        V: (batch_size, ..., seq_len, d_v) s,v
        mask: (batch_size, ..., seq_len, seq_len) s,s
    '''
    d_k = Q.shape[-1]
    presoftmax = torch.einsum("...s k, ...t k -> ...s t", Q, K) / math.sqrt(d_k)
    if mask is not None:
        presoftmax = presoftmax.masked_fill(mask == 0, float("-inf"))
    softmax_output = softmax(presoftmax, dim=-1)
    attention = torch.einsum("...s t, ...t v -> ...s v", softmax_output, V)
    return attention

