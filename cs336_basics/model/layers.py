import torch
import torch.nn as nn
import math
from .utils import scaled_dot_product_attention

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Construct an embedding module.

        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        # initialize your embedding matrix as a nn.Parameter
        # store the embedding matrix with the d_model being the final dimension 
        self.embedding = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        # N(µ = 0, σ2 = 1) truncated at [−3, 3]
        sigma = math.sqrt(1)
        torch.nn.init.trunc_normal_(self.embedding, mean=0, std=sigma, a= -3, b= 3) #  std=1.0 / math.sqrt(in_features))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids: torch.Tensor Input tensor of token IDs

        Returns:
            torch.Tensor: Embedding vectors for the input token IDs
        """
        return self.embedding[token_ids]


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Construct a linear transformation module without bias.
        
        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output 
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__() # super constructor 
        self.in_features = in_features
        self.out_features = out_features
        # construct and store your parameter as W for memory ordering reasons, putting it in an nn.Parameter
        self.W = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        # N(µ = 0, σ2 = 2/(din+dout)) truncated at [−3σ, 3σ] 
        sigma = math.sqrt(2.0/(self.in_features+self.out_features))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=sigma, a= -3*sigma, b= 3*sigma) #  std=1.0 / math.sqrt(in_features))
        self.device = device
        self.dtype = dtype

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        return torch.einsum('...i,ji->...j', x, self.W)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Construct the RMSNorm module.

        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model)) # the parameters values will be loaded into a module with load_state_dict()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) 
        and return a tensor of the same shape.

        Args:
            x: torch.Tensor Input tensor of shape (batch_size, sequence_length, d_model)

        Returns:
            torch.Tensor: Normalized tensor of the same shape
        """
        # Remember to upcast your input to torch.float32 before performing the normalization (and
        # later downcast to the original dtype), as described above.

        in_dtype = x.dtype
        # upcast 
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        def rms(a):
            return torch.sqrt(torch.mean(a**2, dim=-1, keepdim=True) + self.eps)

        result = (x / rms(x) ) * self.gain
        # Return the result in the original dtype (downcast)
        return result.to(in_dtype)
    
class SwiGLU(nn.Module):
    
    def __init__(self, d_model, hidden_size, device=None, dtype=None):
        """
        Construct the SwiGLU module.

        Args:
            d_model: int Hidden dimension of the model, Dimensionality of the feedforward input and output.
            hidden_size: d_ff, Dimensionality of the up-project happening internally to your swiglu

        """
        super().__init__()

        # Initialize parameters with proper shapes
        # w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        # w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        # w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3 
        self.w1 = nn.Parameter(torch.empty((hidden_size, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, hidden_size), device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty((hidden_size, d_model), device=device, dtype=dtype))

        # Add weight attribute to match test expectations
        self.w1.weight = self.w1
        self.w2.weight = self.w2
        self.w3.weight = self.w3

    def forward(self, x):
        '''
            x is of shape (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

            Deliverable: Implement the SwiGLU feed-forward network, composed of a SiLU activation
            function and a GLU.
        ''' 

        # x is of shape (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.
        # SILU activation function (size: (..., hidden_size))
        w1_x = torch.einsum('hd, ...d->...h',self.w1, x)
        silu = torch.sigmoid(w1_x) * w1_x  # x is of shape (..., hidden_size)
        w3_x = torch.einsum('hd, ...d->...h', self.w3, x)

        # element wise product of silu and w3_x
        output = silu * w3_x

        # apply w2 to the output
        return torch.einsum('...h,dh->...d', output, self.w2)  # (..., d_model)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors (d in the handout)
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k # (d in the handout)
        self.max_seq_len = max_seq_len
        self.device = device

        # assert that dimensions are divisible by 2
        assert self.d_k % 2 == 0, "d_k must be divisible by 2"

        # Create position indices (get the value of i in the handout))
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)  # (max_seq_len, 1) 
        # Create frequency indices (get the value of k in the handout, (d_k/2,)
        freqs = torch.arange(0, d_k/2).float()  # (d_k/2,)
        
        # Compute theta values for each position and frequency (get the value of theta_i,k in the handout)
        # dimesions work out bc of broadcasting from right to left 
        # (max_seq_len, 1) / (d_k/2,) --> (max_seq_len, 1) / (1, d_k/2,) -->  (max_seq_len, d_k/2) / (max_seq_len, d_k/2,) -->  (max_seq_len, d_k/2)        
        theta_vals = positions / (self.theta ** (2 * freqs / self.d_k))  # (max_seq_len, d_k/2)

        # Precompute sin and cos values
        sin_vals = torch.sin(theta_vals)  # (max_seq_len, d_k/2)
        cos_vals = torch.cos(theta_vals)  # (max_seq_len, d_k/2)

        # create buffers (persistent=False means they won't be saved in state_dict)
        self.register_buffer('sin_vals', sin_vals, persistent=False)
        self.register_buffer('cos_vals', cos_vals, persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.

        Args:
            x: torch.Tensor Input tensor of shape (..., seq_len, d_k)
            token_positions: torch.Tensor Tensor of shape (..., seq_len) specifying the token positions of
            x along the sequence dimension.

        Returns:
            torch.Tensor: Processed tensor of shape (..., seq_len, d_k)
        """
        # Get sin and cos values for the given token_positions
        sin = self.sin_vals[token_positions]  # (..., seq_len, d_k/2)
        cos = self.cos_vals[token_positions]  # (..., seq_len, d_k/2)
        
        # Split x into pairs of dimensions for rotation every other row of the input 
        x1 = x[..., 0::2]  # take even indices: 0,2,4,...
        x2 = x[..., 1::2]  # take odd indices: 1,3,5,...

        # Apply rotation
        # x_rotated = [x1 * cos - x2 * sin, x1 * sin + x2 * cos] 
        # then concatenate the results for so that it makes it every other row 
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        # Interleave the rotated values in every other row
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x1 * cos - x2 * sin  # Even indices
        x_rotated[..., 1::2] = x1 * sin + x2 * cos  # Odd indices

        return x_rotated
       
class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None = None, theta: float | None = None, in_features: torch.Tensor | None = None, token_positions: torch.Tensor | None = None, device=None, dtype=None):
        '''
        Implement causal multi-head self-attention as a torch.nn.Module. 
    Args: 
        d_model: int Dimensionality of the Transformer block inputs.
        num_heads: int Number of heads to use in multi-head self-attention.
        max_seq_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model//num_heads
        self.dv = self.dk
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.in_features = in_features
        self.token_positions = token_positions
        self.device = device
        self.dtype = dtype

        # define linear layers for q, k, v, and o
        # WQ ∈ ℝ^(h*dk × dmodel)
        self.q_proj = Linear(self.d_model, self.num_heads * self.dk, device=device, dtype=dtype)
        # WK ∈ ℝ^(h*dk × dmodel)
        self.k_proj = Linear(self.d_model, self.num_heads * self.dk, device=device, dtype=dtype)
        # WV ∈ ℝ^(h*dv × dmodel)
        self.v_proj = Linear(self.d_model, self.num_heads * self.dv, device=device, dtype=dtype)
        # WO ∈ ℝ^(dmodel × h*dv) 
        self.o_proj = Linear(self.num_heads * self.dv, self.d_model, device=device, dtype=dtype)

        # only apply rope if some token_positions were passed in
        if token_positions is not None: 
            # self.rope = RotaryPositionalEmbedding(self.theta, self.dk, self.max_seq_len, device=self.device)
            self.rope = RotaryPositionalEmbedding(self.theta, self.d_model // self.num_heads, self.max_seq_len, device=self.device)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        '''
        Process an input tensor of shape (..., seq_len, d_model) and return a tensor of the same shape.
        '''
        # Project input to Q, K, V
        q = self.q_proj(x)  # (..., seq_len, h*dk)
        k = self.k_proj(x)  # (..., seq_len, h*dk)
        v = self.v_proj(x)  # (..., seq_len, h*dv)

        # split the last dimension to separate heads, then transpose so that the head dimension is first
        q = q.view(*q.shape[:-1], self.num_heads, self.dk).transpose(1, 2)  # (..., seq_len, h, dk)
        k = k.view(*k.shape[:-1], self.num_heads, self.dk).transpose(1, 2)  # (..., seq_len, h, dk)
        v = v.view(*v.shape[:-1], self.num_heads, self.dv).transpose(1, 2)  # (..., seq_len, h, dv)

        # Apply RoPE if needed
        if self.rope is not None and token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # mask part 
        seq_len = x.shape[-2] # n in the handout i think
        mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        x = scaled_dot_product_attention(q, k, v, mask)
        x = x.transpose(1, 2)
        x = x.reshape(*x.shape[:-2], -1)  # (..., seq_len, h*dv)

        # apply o_proj
        return self.o_proj(x)


    



class TransformerBlock(nn.Module):
    '''
    Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure 2. Your
    Transformer block should accept (at least) the following parameters.
    d_model: int Dimensionality of the Transformer block inputs.
    num_heads: int Number of heads to use in multi-head self-attention.
    d_ff: int Dimensionality of the position-wise feed-forward inner layer
    max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
    theta (float): RoPE parameter.
    
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        # RMSNorm 1 
        self.rms1 = RMSNorm(d_model)
        # MultiheadSelfAttention 
        # TODO: with ROPE (PUT IN ROPE WHEN IT WORKS)
        # max_seq_len, theta, token_positions=None, None, None
        self.mha = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        # RMSNorm 2
        self.rms2 = RMSNorm(d_model)
        # position wise feed forward network
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        '''
        Process an input tensor of shape (..., seq_len, d_model) and return a tensor of the same shape.
        '''
        # apply rms1 and mha
        y = x+self.mha(self.rms1(x), token_positions)
        # apply rms2 and ffn
        output = y + self.ffn(self.rms2(y))
        return output