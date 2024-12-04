import math
import torch
import torch.nn.functional as F

from ..utils import inspect, normalize


def bmm(x, w):
    """ assuming x, w: (N, B, C, H, W), perform batch matmul."""
    x = x.transpose(0, 1)
    B, N, C, H, W = x.shape
    x = x.reshape(B, N, -1)
    return torch.bmm(
        w.transpose(-1, -2),
        x
    ).reshape(B, N, C, H, W).transpose(0, 1)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight.abs(), dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True).to(value.dtype)
    return attn_weight @ value


#TODO try having another state and then let them evolve together.. coupled
## like. each giving differential to each other. but with a slight asymmetry,
## perhaps can help prevent mode collapse. think maxwells eqns

class CAN:
    """
    A graph where the nodes are images, and the edges are attention
    connections. Updates through time as a diffeq.

    Design decisions:
        how to parameterize the network topology? do predefined? or let some
            part of the state determing the network topology. almost certainly
            the second. i suppose included in the network topology is not only
            the edges, but the edge weights, which key, query, and value they
            use
                    .. oohhh can really just use a single controller node, and use
                    different channels in it to read off the kernel and the
                    weight matrix.. i suppose this requires channels > 1 .. 
        how to parameterize the step size. presumably want the step size to be
        different in different parts of the network. there is some chance that
        simply a difference in ordero of magnitude of the states will provide
        this difference in inertia that we are looking for, but i kinda feel
        like it could be better to explicitly give the model the ability to condition their ideas 

        i suppose need to ensure there is mixing across the channel dim.. i
        suppose one way of testing this would be trying to reconstruct a red
        b/w image as a blue b/w image!

        
    """
    def __init__(
        self,
        B: int,
        C: int,
        H: int,
        W: int,
        N: int,
        wei_idx: int = 0,
        query_idx: int = None,
        key_idx: int = 2,
        value_idx: int = 2,
        stdin_idx: int = 4,
        stdout_idx: int = 5,
        use_complex: bool = True,
        clip_min: float = -1,
        clip_max: float = 1,
        verbose: bool = False,
    ):

        self.verbose = verbose

        def p(*args):
            if self.verbose:
                print(*args)

        self.N = N
        self.B = B
        self.C = C
        self.H = H
        self.W = W

        self.clip_min = clip_min
        self.clip_max = clip_max


        self.rand = torch.randn ## TODO check what happens if we use uniform (torch.rand) instead of normal. is there any qualitative difference? what about other initializations?

        self.complex = use_complex
        self.dtype = torch.cfloat if self.complex else torch.float

        ## TODO test which mem layout is more efficient: (N, B, ...) or (B, N, ...). for now using (N, B, ...) for ease of indexing into specific nodes..
        self.state = self.rand(N, B, C, H, W, dtype=self.dtype)
        self.proj_to_torus()

        self.losses = torch.zeros(B)

        self.wei_idx = min(wei_idx, N - 1)
        # self.ker_idx = min(ker_idx, N - 1)
        self.query_idx = query_idx
        self.key_idx = key_idx
        self.value_idx = value_idx
        self.stdout_idx = min(stdout_idx, N - 1)
        self.stdin_idx = min(stdin_idx, N - 1)

        self.color = (C >= 3)

    def output(self, node_idx=None):
        """ output from self.state[node_idx], ready for matplotlib.
        """
        if node_idx is None:
            node_idx = self.stdout_idx

        if self.color:
            out = self.state[node_idx, :, 0:3, :, :] # just use first three color channels for display for now
        else:
            out = torch.mean(self.state[node_idx], dim=1).unsqueeze(1) # mean across channel dim to get b/w image
        # out.shape = (B, C_out, H, W)
        out = out.detach().cpu().permute(0, 2, 3, 1) # (B, C_out, H, W) -> (B, H, W, C_out)

        out = out.real.abs() ## take the absolute value of the real part of out
        ## out = torch.frac(1 + out) # pass to [0, 1] pixel values .. new impl should already have pixel values in [0, 1]

        return out


    def input(self, inp, node_idx=None, alpha=1, ga_eval=False):
        """ update state[idx] as a convex combination with inp."""
        if node_idx is None:
            node_idx = self.stdin_idx

        H, W, C = inp.shape

        inp = inp.unsqueeze(0) # (H, W, C) -> (1, H, W, C)

        if ga_eval:
            # evaluate each batch (in ga language: compute the fitness of the various candidate solutions)
            with torch.no_grad():
                preds = self.output(self.stdout_idx)
                loss = F.mse_loss(preds, inp.expand(*preds.shape), reduction='none') # elementwise mse loss
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) # average all but batch dim
                self.losses += loss

        inp = inp.permute(0, 3, 1, 2) # (1, H, W, C) -> (1, C, H, W)

        if self.C >= C: # if more channels in self.state than in inp
            self.state[node_idx][:, 0:C, :, :] = alpha * inp  +  (1 - alpha) * self.state[node_idx][:, 0:C, :, :]
        elif self.C == 1:
            inp = torch.mean(inp, dim=1) # (1, C, H, W) -> (1, 1, H, W)
            self.state[node_idx] = alpha * inp  +  (1 - alpha) * self.state[node_idx]
        else:
            raise Exception('TODO have not handled case where (self.C != 1) and (C > self.C)')


    def add_noise(self, noise_scale: float, batch_dim: int = None):
        self.state += noise_scale * torch.randn_like(self.state)
        self.proj_to_torus()

    def ga_step(self, k, max_noise_scale):
        """ Run a genetic algorithm step.
            This looks like:
                -sort batches by prediction error
                -select top-k
                -copy them along the batch dimension (i.e. have k divides B,
                    and populate batch dim with k * (B//k) copies of the topk)
                -TODO include some random sampling. for right now just going to
                    do determinsitc, but really we want to be sampling from some
                    distribution like a poisson or something.
                -reset losses (TODO might not want to do this, but rather have
                    some sort of running average or something, i mean we really do
                    want to eventually be capturing long horizon dynamics..)

        """
        N, B, C, H, W = self.state.shape

        assert (self.B % k) == 0, 'k must divide B currently, for simplicity of implementation'
        ranking = torch.argsort(self.losses)
        keepers = self.state[:, ranking[:k]]
        keepers = keepers.repeat(1, self.B // k, 1, 1, 1)

        noise_scale = torch.linspace(0, max_noise_scale, self.B // k).reshape(-1, 1)
        noise_scale = noise_scale.expand(self.B // k, k).reshape(1, self.B, 1, 1, 1)

        noise = torch.randn_like(self.state) * noise_scale

        self.state = keepers + noise

        self.losses = torch.zeros_like(self.losses) # reset losses. TODO think: do we want some sort of moving average, or some other way of keeping track of expectation over time .. this is probably a question to be answered with some thinking from RL ..


    def wei(self):
        wei = self.state[self.wei_idx] # (B, C, H, W)
        ## if self.complex: wei = wei.real # just take the real part
        wei = torch.mean(wei, dim=1).unsqueeze(1) # (B, C, H, W) -> (B, 1, H, W)
        if self.complex:
            wei = torch.complex(
                F.interpolate(wei.real, size=(self.N, self.N)).squeeze(1), # (B, H, W) -> (B, N, N)
                F.interpolate(wei.imag, size=(self.N, self.N)).squeeze(1) # (B, H, W) -> (B, N, N)
            )
        else:
            wei = F.interpolate(wei, size=(self.N, self.N)).squeeze(1) # (B, H, W) -> (B, N, N)
        return wei


    def ker(self, control_node_idx):
        if control_node_idx is None:
            return self.state

        k = self.state[control_node_idx] # (B, C, H, W)                                      ## get the node that parameterizes the kernels
        if self.complex: k = k.real
        k = torch.mean(k, dim=1).unsqueeze(1) # (B, C, H, W) -> (B, 1, H, W)             ## make into a b/w image
        # k = F.interpolate(k, size=(self.N, self.N)).squeeze(1) # (B, H, W) -> (B, N, N)  ## make correct shape
        k = F.interpolate(k, size=(self.N, self.N), mode='bilinear', antialias=True).squeeze(1) # (B, H, W) -> (B, N, N)  ## make correct shape
        k = torch.argmax(k, dim=1) # (B, N, N) -> (B, N)  ## now get indices             ## obtain batched list of indices, N indices per batch
        state = self.state.transpose(0, 1) # (N, B, ...) -> (B, N, ...)                  ## put batch dim first in state, advanced indexing expects this
        batch_idxs = torch.arange(self.B).unsqueeze(-1).expand(self.B, self.N) # (B, N)  ## generate batch indices
        ker = state[batch_idxs, k] # (B, N, H, W, C)                                     ## index into state to get a kernel for each node, for each batch dim
        ker = ker.transpose(0, 1) # (B, N, ...) -> (N, B, ...)                           ## put node dim first in state (maintain consistency with how the rest of this class is currently implemented with self.state.shape = (N, B, C, H, W)
        return ker


    def proj_to_torus(self):
        if self.complex:
            ## self.state = self.state / (self.state.abs() + 1e-9) # only care about the phase angle of each coordinate
            self.state = torch.complex(torch.frac(self.state.real), torch.frac(self.state.imag)) ## or other parameterization of a complex torus ## TODO experiment with these two different projections onto a complex tori

        else:
            # some kinda hacky coersion of a real tensor to [0, 1]^self.state.numel -- hacky since multiplication really doesnt play nice with this projection (observe $(1+a)*b = ab + b \neq ab$ (mod 1)) (i.e. this map onto the torus is not a homomorphism for real multiplication .. or something like that!)
            self.state = torch.frac(1 + torch.frac(self.state))


    def step(self, alpha=0.01):
        """
        update: X += alpha * dXdt

        ## TODO make alpha be something more interesting than just slider
        controlled probably.. lets let the network parameterize this i think.
        or potentially, when building more hand cratfted architectures, would
        presumably like to be able to control the 'inertia' (alpha) of the
        various nodes.. 
        """

        def p(*args):
            if self.verbose:
                print(*args)

        ## TODO come up with a principled way of doing the step ..
        ## TODO think about doing some learned linear transformations, i.e. with mha or something..


        # ker = self.ker(self.ker_idx) # (N, B, C, H, W)

        # # TODO observe that here there is no communication across channels .. i.e. C is treated as just another batch dim here ..
        # sk = self.state @ ker.transpose(-2, -1)

        # attn
        query = self.ker(self.query_idx)
        key = self.ker(self.key_idx)
        value = self.ker(self.value_idx)

        query = query.reshape(self.N, self.B, -1).transpose(0, 1)
        key = key.reshape(self.N, self.B, -1).transpose(0, 1)
        value = value.reshape(self.N, self.B, -1).transpose(0, 1)

        scale_factor = 1 / math.sqrt(query.size(-1))

        attn = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            scale=scale_factor,
        ).transpose(0, 1).reshape(self.N, self.B, self.C, self.H, self.W)

        dxdt = torch.zeros_like(self.state)

        dxdt += attn
        ## TODO pass in a hyperparam whether or not to use wei .. for now just commenting out.. doesnt appear (visually) to add to expressivity. theoretically i feel like it should help but idk.. perhaps its a question of introducing a scaling factor like in SDPA .. in fact i think this would work well ..
        ## TODO add a scaling param ..
        wei = self.wei() # (B, N, N)
        # dxdt += bmm(attn, wei) * scale_factor # flow sk along adjacency structure given by wei .. currently untrained, but seems like uncommenting this line gives somewhat less interesting behavior. of course, really should implement the ga first and then make judgements about expressivity after a bit of selection

        self.state += alpha * dxdt

        self.proj_to_torus()





    ## deprecated

    ## ## TODO pass in which activation we're using in command line args. will be better for methodical develepoment .. and repeataable experimentation
    ## def activation(self):
    ##     """ currently feels fairly unprincipled how i am doing with this activation. curently restricting values to [0, 1], yay its an nd-torus..
    ##     """
    ##     # self.state = torch.frac(self.state)
    ##     # self.state = torch.frac(1 + torch.frac(self.state))
    ##     self.state = torch.tanh(self.state)
    ##     # self.state = torch.clip(self.state, self.clip_min, self.clip_max)
    ##     # self.state = normalize(self.state) ## uninteresting results with this approach it looks like
    ##     # self.state = self.state / torch.linalg.norm(self.state, dim=TODO) ## try this .. feels like this could be a good approach, and amenable having self.state.dtype = torch.complex
    ##                                                                         ## TODO figure out exaclty what batch norm and layer norm are doing .. seems like these are methods that achieve a similar goal and have had success .. idk batch norm required different behavior for train time and test time, so dont like that ..
    ##                                                                         ## layer norm seems better.. in the paper they mention that it is an approach that works for rnns ..
    ##     # observe normalization should make a bit more sense when think of complex valued state ..
