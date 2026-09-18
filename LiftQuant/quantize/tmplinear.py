import torch
import torch.nn as nn
import torch.nn.functional as F
from trans_utils import Hadamard_trans, SVDDecomposeTransMatrix, HalfSVDDecomposeTransMatrix, GHalfSVDDecomposeTransMatrix
from quantize.int_linear import *
from gptq.quant import Quantizer

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os 
from tqdm import tqdm
from quantize.E8Q import *
from trans_utils import *
from itertools import combinations
from torch.utils.checkpoint import checkpoint

from itertools import product
import math
def plot_box_plot(datalist, figname, titlelist, limit=0.5):

    # 创建并排的子图
    if len(datalist) ==1:
        fig, axes = plt.subplots(1, 1, figsize=(30, 10))
    elif len(datalist) ==2:
        fig, axes = plt.subplots(1, 2, figsize=(30, 10))
    elif len(datalist) ==3:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    elif len(datalist) ==4:
        fig, axes = plt.subplots(1, 4, figsize=(40, 10))
    # 绘制箱线图
    for i, data in enumerate(datalist):
        q1 = torch.quantile(data, 0.25, dim=0).to('cpu').numpy()
        q3 = torch.quantile(data, 0.75, dim=0).to('cpu').numpy()
        p1 = torch.quantile(data, 0.01, dim=0).to('cpu').numpy()
        p99 = torch.quantile(data, 0.99, dim=0).to('cpu').numpy()
        min = torch.min(data, dim=0)[0].to('cpu').numpy()
        max = torch.max(data, dim=0)[0].to('cpu').numpy()

        for j in range(data.shape[1]):
            # 第一个胡须（到最小值和最大值）
            axes[i].plot([j+1, j+1], [min[j], max[j]], color="brown", linewidth=1)
            
            # 第二个胡须（到99%分位数）
            axes[i].plot([j+1, j+1], [p1[j], p99[j]], color="orange", linewidth=1)
            
            # 第三个胡须（到25%分位数）
            axes[i].plot([j+1, j+1], [q1[j], q3[j]], color="gray", linewidth=1)

        axes[i].set_title(titlelist[i], fontsize=16)
        if i ==0:
            axes[i].set_ylabel('Activation Value', fontsize=14)
        axes[i].set_xlabel('Channel Index', fontsize=14)
        #axes[i].set_xticklabels([]) 
    
    

    # 添加图例
    
    legend_elements = [
        Line2D([0], [0], color='gray', lw=4, label='25%-75% Percentile'),
        Line2D([0], [0], color='orange', lw=4, label='1%-99% Percentile'),
        Line2D([0], [0], color='brown', lw=4, label='Min-Max')
    ]
    

    plt.tight_layout()
    os.makedirs(os.path.dirname(figname), exist_ok=True)
    plt.savefig(figname)
    plt.close()

    return 0 

class MoeSharedRotScale(nn.Module):
    """Group-level shared rotation + scaling for MoE experts.

    One instance is created per ``(group, projection_type)`` and owns the
    shared ``Trans`` / ``a1`` / ``a2``.  Expert ``TmpLinear`` / ``FWTLinear``
    modules only *reference* these objects (via ``object.__setattr__``) so they
    are registered exactly once in ``state_dict()``.
    """

    def __init__(self, ic, expc, training_trans, groupsize):
        super(MoeSharedRotScale, self).__init__()
        self.ic = ic
        self.training_trans = training_trans
        self.groupsize = groupsize

        parts = expc.split('to')
        self.root = int(parts[1])
        self.root2 = int(parts[0])
        if self.root == 8:
            self.transdim2 = 128 if ic > 10000 else 64
        else:
            self.transdim2 = 64
        self.transdim1 = math.ceil(ic / self.transdim2)
        self.expic = self.transdim1 * self.transdim2

        # Keep the 2-D layout used by TmpLinear.find_params/quant_tmpweight.
        self.a1 = nn.Parameter(torch.ones(self.transdim1, self.transdim2))
        self.a2 = nn.Parameter(torch.ones(self.transdim1, self.transdim2 // self.root))

        if training_trans:
            self.Trans = HalfSVDDecomposeTransMatrix(self.transdim1, self.transdim2)
        else:
            self.Trans = None


class FWTLinear(nn.Module):
    def __init__(
        self,
    ):
        super(FWTLinear, self).__init__()

    def convert_form_tmplinear(
        self,
        tmp_module,
        expc = 'n',
        training_trans = False,
        bits = 2,
        groupsize = -1,
        fast_nearest = True,
        shared = None,
        to_buffer = True,
    ):
        self.maxq = 2**bits-1
        #print(self.maxq)
        self.training_trans = training_trans
        self.groupsize = groupsize
        self.adaquant = True
        self.oc, self.ic =  tmp_module.oc, tmp_module.ic
        
        parts = expc.split('to')
        self.root = int(parts[1])
        self.root2 = int(parts[0])
        if self.root == 8:
            if self.ic >10000:
                self.transdim2 = 128
            else:
                self.transdim2 = 64
        #self.transdim2 = round(math.sqrt(self.ic)/self.root)*self.root
        self.transdim1 = math.ceil(self.ic/self.transdim2)
        self.expic = self.transdim1 * self.transdim2
        
        self.fast_nearest = fast_nearest
        self.expc = expc

        self.bias = tmp_module.bias
        with torch.no_grad():
            if shared is not None:
                # Group-shared rotation/scaling: reference the holder-owned objects
                # instead of registering per-expert copies.
                self.register_parameter('scale', nn.Parameter(tmp_module.quantizer.scale.data*2*F.sigmoid(tmp_module.quantizer.alpha.data)))
                object.__setattr__(self, 'a1', shared.a1)
                object.__setattr__(self, 'a2', shared.a2)
                object.__setattr__(self, 'Trans', shared.Trans)
                object.__setattr__(self, 'shared', shared)

                self.weight = tmp_module.orilinear.weight.reshape(self.oc, self.transdim1, -1) * shared.a1.data
                if self.training_trans:
                    # Trans is still in training mode here; the caller converts it to
                    # buffer (to_buffer) exactly once per group afterwards.
                    self.weight = self.Trans(self.weight)
                else:
                    self.weight = Hadamard_trans(self.weight, self.transdim1, self.transdim2)
                self.weight = self.weight * shared.a2.data.repeat_interleave(self.root, dim=-1)
                self.weight = self.weight.reshape(self.oc, -1)
                weight = self.weight
                del self.weight
                self.register_parameter('weight', nn.Parameter(weight))
            else:
                self.register_parameter('scale', nn.Parameter(tmp_module.quantizer.scale.data*2*F.sigmoid(tmp_module.quantizer.alpha.data)))
                self.register_parameter('a1', nn.Parameter(self.clamp_ste(tmp_module.a1.data, 0.02, 50)) )
                #self.register_parameter('a2', nn.Parameter( (tmp_module.a3.data.reshape(-1)/tmp_module.a2.data.reshape(-1))))
                self.register_parameter('a2', nn.Parameter( 1./tmp_module.a2.data.reshape(-1)))

                self.weight = tmp_module.orilinear.weight.reshape(self.oc, self.transdim1, -1)  * self.a1.data

                if self.training_trans:
                    self.Trans = tmp_module.Trans
                    self.weight = self.Trans(self.weight)
                    if to_buffer:
                        self.Trans.to_buffer()
                else:
                    self.weight = Hadamard_trans(self.weight, self.transdim1, self.transdim2)
                self.weight = self.weight * tmp_module.a2.data.repeat_interleave(self.root, dim=-1)
                self.weight = self.weight.reshape(self.oc, -1)
                weight = self.weight
                del self.weight
                self.register_parameter('weight', nn.Parameter(weight))
                self.a1.data = self.a1.data.reshape(-1)
        self.packed_flag = False

    def bit_channel_convert(self, fast=False):
        # convert 8x2bits to 16x1bits
        if self.weight.dtype == torch.float16:
            minmin = 1e-4
            maxmax = 1e4
        else:
            minmin = 1e-7
            maxmax = 1e7
        l2 = torch.clamp(self.weight.pow(2).mean(dim=-1,keepdim=True).pow(0.5), minmin, maxmax)
        norm_weight = self.weight/l2
        norm_weight = norm_weight.reshape(-1,self.root)
        device = self.weight.device
        
        M_path = './lattice/' + self.expc +'.pt'
        T = torch.load(M_path).to(device)

        if fast:   # use to generate null weight in e2e finetune
            alpha = self.root2/self.root
            qnorm_weight = torch.randn(self.weight.shape[0], int(self.weight.shape[1]*alpha)).to(device)
        else:
            if self.fast_nearest:
                qnorm_weight = self.find_nearest_fast(norm_weight, T, self.root2-self.root, batch_size=128, device=device)
            else:
                codes = torch.empty((2**self.root2, self.root2), dtype=torch.float32, device=device)
                for i in range(self.root2):
                    # 周期长度：2^(i+1)
                    repeat_len = 2 ** (i + 1)
                    block = torch.cat([torch.full((2**i,), -1.), torch.full((2**i,), 1.)]).to(self.weight)
                    reps = 2**self.root2 // repeat_len
                    codes[:, i] = block.repeat(reps)
                points = codes@T.t()
                indices = self.find_nearest_in_batches(norm_weight, points, batch_size=1024)    # (N,)
                qnorm_weight = codes[indices] 
            print('related std error in trans domain',(qnorm_weight@T.t() - norm_weight).std()/ norm_weight.std())
        qnorm_weight = qnorm_weight.reshape(l2.shape[0],-1)
    
        ow = self.get_oriweight()
        oqw = self.get_oldqweight()
        qw = ((qnorm_weight*l2).reshape(-1,self.root2)@T.t()).reshape(l2.shape[0],-1)
        if fast==False:
            print('std error in ori domain', (self.weight-qw).std(), self.weight.std())
        del self.weight
        self.register_parameter('weight', nn.Parameter(qnorm_weight * l2 ))
        
        self.maxq = 1
        self.scale.data = l2*2. 
        M = torch.zeros(self.transdim2//self.root * self.root2, self.transdim2).to(T)
        for i in range(self.transdim2//self.root):
            row_start = i * self.root2
            col_start = i * self.root
            M[row_start:row_start+self.root2, col_start:col_start+self.root] = T.t()

        a2 = self.a2.data.repeat_interleave(self.root2, dim=-1)       
        del self.a2
        self.register_parameter('a2', nn.Parameter(a2.to(self.weight)))
        self.root = 1
        mr = self.Trans.linear_right.data.to(device)
        del self.Trans.linear_right 

        self.Trans.register_parameter('linear_right', nn.Parameter(M@mr))
        if fast == False:
            print('related std error in ori domain', (self.get_weight()-ow).std()/ow.std())
            print('related std error in ori domain, Uniform Quantizater', (oqw-ow).std()/ow.std())

    def prepare_shared_search(self):
        # Per-expert part of the vector-quantization conversion for grouped MoE.
        # Computes the per-row L2 norm and normalized weight used by the lattice
        # nearest-neighbour search. Bit-identical to the inline logic formerly
        # inside bit_channel_convert_shared().
        if self.weight.dtype == torch.float16:
            minmin = 1e-4
            maxmax = 1e4
        else:
            minmin = 1e-7
            maxmax = 1e7
        l2 = torch.clamp(self.weight.pow(2).mean(dim=-1,keepdim=True).pow(0.5), minmin, maxmax)
        norm_weight = self.weight/l2
        norm_weight = norm_weight.reshape(-1,self.root)
        return l2, norm_weight

    def finalize_shared_search(self, qnorm_weight, l2):
        qnorm_weight = qnorm_weight.reshape(l2.shape[0],-1)

        del self.weight
        self.register_parameter('weight', nn.Parameter(qnorm_weight * l2 ))

        self.maxq = 1
        self.scale.data = l2*2.
        self.root = 1

    def bit_channel_convert_shared(self, fast=False, cache=None):
        # Per-expert part of the vector-quantization conversion for grouped MoE.
        # Group-shared mutations (a2 root2-expansion and Trans.linear_right
        # composition) are performed once per group by
        # prepare_moe_shared_for_export().
        l2, norm_weight = self.prepare_shared_search()
        device = self.weight.device

        if cache is not None:
            T = cache['T']
        else:
            M_path = './lattice/' + self.expc +'.pt'
            T = torch.load(M_path).to(device)

        if fast:   # use to generate null weight in e2e finetune
            alpha = self.root2/self.root
            qnorm_weight = torch.randn(self.weight.shape[0], int(self.weight.shape[1]*alpha)).to(device)
        else:
            if self.fast_nearest:
                qnorm_weight = self.find_nearest_fast(norm_weight, T, self.root2-self.root, batch_size=128, device=device, cache=cache)
            else:
                codes = torch.empty((2**self.root2, self.root2), dtype=torch.float32, device=device)
                for i in range(self.root2):
                    # 周期长度：2^(i+1)
                    repeat_len = 2 ** (i + 1)
                    block = torch.cat([torch.full((2**i,), -1.), torch.full((2**i,), 1.)]).to(self.weight)
                    reps = 2**self.root2 // repeat_len
                    codes[:, i] = block.repeat(reps)
                points = codes@T.t()
                indices = self.find_nearest_in_batches(norm_weight, points, batch_size=1024)    # (N,)
                qnorm_weight = codes[indices]
            print('related std error in trans domain',(qnorm_weight@T.t() - norm_weight).std()/ norm_weight.std())
        self.finalize_shared_search(qnorm_weight, l2)

    def find_nearest_in_batches(self, A, points, batch_size=128):
        N = A.shape[0]
        M = points.shape[0]
        indices_list = []
        with torch.no_grad():
            x_sq = (points ** 2).sum(dim=1, keepdim=True).T  #[1, N]
            for i in tqdm(range(0, N, batch_size)):
                A_chunk = A[i:i+batch_size]                           # (bs, 8)
                # 距离计算: (bs, M)
                #dist2 = ((A_chunk[:, None, :] - points[None, :, :])**2).sum(dim=2)
                
                # 3) 计算距离矩阵
                z_sq = (A_chunk ** 2).sum(dim=1, keepdim=True)  # (bs, 1)
                dists = z_sq + x_sq - 2 * A_chunk @ points.T  # (bs, N)

                idx = dists.argmin(dim=1)                             # 最近格点索引
                indices_list.append(idx)
        return torch.cat(indices_list, dim=0)

    def find_nearest_fast(self, W, M, padding_length, batch_size=128, device='cuda', cache=None):
        M = M.to(W).to(torch.float32)
        # 预计算零空间基和逆矩阵（对于固定的M，这些是不变的）
        D_out, D_in = M.shape
        with torch.no_grad():
            if cache is not None:
                M_square_inv = cache['M_square_inv']
                padding_vectors = cache['padding_vectors']
                num_candidates = cache['num_candidates']
            else:
                try:
                    _, _, Vh = torch.linalg.svd(M)
                    N = Vh[D_out:]
                    M_square = torch.cat([M, N], dim=0)
                    M_square_inv = torch.linalg.inv(M_square)
                except torch.linalg.LinAlgError:
                    print("Benchmark SVD/inv failed. M might be singular.")
                    return float('inf')
                    
                # 预计算 padding 向量
                num_candidates_exp = padding_length
                padding_vectors = torch.tensor(
                    list(product([-1, 1], repeat=num_candidates_exp)), 
                    dtype=torch.float32, device=device
                )
                num_candidates = padding_vectors.shape[0]
        encoded_vectors_list = []
        vectors_to_encode = W.view(-1, D_out)
        num_vectors = vectors_to_encode.shape[0]
        for start in tqdm(range(0, num_vectors, batch_size)):
            end = min(start + batch_size, num_vectors)
            
            with torch.no_grad():
                z_batch = vectors_to_encode[start:end]
                bs = z_batch.shape[0]
                # --- 生成候选集 (与训练时逻辑相同) ---
                z_expanded = z_batch.unsqueeze(1).expand(-1, num_candidates, -1)
                paddings_expanded = padding_vectors.unsqueeze(0).expand(bs, -1, -1)
                Y_subset = torch.cat([z_expanded, paddings_expanded], dim=2)
                Y_subset = Y_subset @ M_square_inv.T
                Y_subset = torch.sign(Y_subset)
                
                # --- 在候选集中寻找真正的最近邻 ---
                grid_points_subset = Y_subset @ M.T
                z_expanded_for_dist = z_batch.unsqueeze(1)
                dist_sq_matrix_subset = torch.sum((z_expanded_for_dist - grid_points_subset) ** 2, dim=2)
                
                # 找到每个 z 的最近邻
                _, nn_idx = torch.min(dist_sq_matrix_subset, dim=1)

                # ------ 从 Y_subset 中选出最终的码字 ------
                # nn_idx: (bs,) -> (bs, 1, 1) -> (bs, 1, D_in)
                # 使用 gather 从 Y_subset 中精确地挑选出每个向量对应的最佳码字
                best_y = torch.gather(Y_subset, 1, nn_idx.view(-1, 1, 1).expand(-1, 1, D_in)).squeeze(1)
                
                encoded_vectors_list.append(best_y)
            
        W_encoded_flat = torch.cat(encoded_vectors_list, dim=0)
        return W_encoded_flat

    def round_ste(self, x: torch.Tensor):
        """
        Implement Straight-Through Estimator for rounding operation.
        """
        return (x.round() - x).detach() + x

    def Hadamard_trans(self, data, dim1, dim2, inv):
        data_shape = data.shape 
        data=data.reshape(-1,dim1,dim2)
        H1 = self.H1
        H2 = self.H2
        if inv:
            H1 = H1.T
            H2 = H2.T
        H1 = H1.to(data)
        H2 = H2.to(data)
        data = H1@data@H2
        return data.reshape(data_shape)

    def clamp_ste(self, x: torch.Tensor, min, max):
        return (x.clamp(min,max) - x).detach() + x
    
    def _a1(self):
        shared = getattr(self, 'shared', None)
        return shared.a1 if shared is not None else self.a1

    def _a2(self):
        shared = getattr(self, 'shared', None)
        return shared.a2 if shared is not None else self.a2

    def _trans(self):
        shared = getattr(self, 'shared', None)
        return shared.Trans if shared is not None else self.Trans

    def get_oldqweight(self):
        if True:
            if self.groupsize == -1:
                scale = torch.clamp(self.scale, 1e-6, 1e6)
                weight = (torch.clamp(self.round_ste(self.weight/scale+self.maxq/2), 0, self.maxq) - self.maxq/2) * self.scale 
            else:
                shape = self.weight.shape
                weight = (torch.clamp(self.round_ste(self.weight.reshape([-1,160])/scale+self.maxq/2), 0, self.maxq) - self.maxq/2) * self.scale 
                weight = weight.reshape(shape)
            
            weight = weight * self._a2().repeat_interleave(self.root, dim=-1)  
            if self.training_trans:
                weight = self._trans()(weight, True)
            else:
                weight = Hadamard_trans(weight, dim1= self.transdim1, dim2=self.transdim2, inv = True)
            weight = weight.reshape(self.oc, self.transdim1, self.transdim2)[:,:,:self.expic//self.transdim1]
            weight = weight.reshape(self.oc, self.expic)/self._a1().reshape(-1)
        return weight

    def miniFunction1(self,x, scale):
        return  x * scale 
    def miniFunction2(self,x, a2):
        return  x * a2.repeat_interleave(self.root, dim=-1)  
    def miniFunction3(self,x, a1):
        return  x.reshape(self.oc, self.expic)/a1

    def get_weight(self):
        if True:
            if self.groupsize == -1:
                if self.packed_flag:
                    weight = (self.unpack_bits_uint8(self.packed_weight).reshape(self.oc, -1)- self.maxq/2)*self.scale
                else:
                    scale = torch.clamp(self.scale, 1e-7, 1e7)
                    weight = (torch.clamp(self.round_ste(self.weight+self.maxq/2), 0, self.maxq) - self.maxq/2) * self.scale 
                    #weight = checkpoint(self.miniFunction1, weight, self.scale)
            else:
                shape = self.weight.shape
                weight = (torch.clamp(self.round_ste(self.weight.reshape([-1,160])/scale+self.maxq/2), 0, self.maxq) - self.maxq/2) * self.scale 
                weight = weight.reshape(shape)
            
            weight = weight * self._a2().repeat_interleave(self.root, dim=-1)   #checkpoint(self.miniFunction2, weight, self.a2)
            if self.training_trans:
                weight = self._trans()(weight, True) #checkpoint(self.Trans, weight, True) # 
            else:
                weight = Hadamard_trans(weight, dim1= self.transdim1, dim2=self.transdim2, inv = True)
            weight = weight.reshape(self.oc, self.transdim1, self.transdim2)[:,:,:self.expic//self.transdim1]
            weight = weight.reshape(self.oc, self.expic)/self._a1().reshape(-1)#checkpoint(self.miniFunction3, weight, self.a1)
            
        return weight
    
    def get_oriweight(self):
        if True:
            weight = self.weight * self._a2().repeat_interleave(self.root, dim=-1)  
            if self.training_trans:
                weight = self._trans()(weight, True)
            else:
                weight = Hadamard_trans(weight, dim1= self.transdim1, dim2=self.transdim2, inv = True)
            weight = weight.reshape(self.oc, self.transdim1, self.transdim2)[:,:,:self.expic//self.transdim1]
            weight = weight.reshape(self.oc, self.expic)/self._a1().reshape(-1)
        return weight

    '''def unpack_bits_uint8(self, packed: torch.Tensor):
        bits = torch.stack([(packed >> i) & 1 for i in range(8)], dim=-1)
        return bits.view(packed.shape[0], -1)'''
    def unpack_bits_uint8(self, packed: torch.Tensor):
        # 1. 创建掩码: [1, 2, 4, 8, 16, 32, 64, 128]
        # 这一步可以作为类的常量预先定义好，避免每次调用都创建
        mask = 2 ** torch.arange(8, dtype=packed.dtype, device=packed.device)
        
        # 2. 利用广播机制解包
        # packed.unsqueeze(-1) 形状变为 [N, 1]
        # mask 形状为 [8]
        # 两者位与运算后形状变为 [N, 8]
        return ((packed.unsqueeze(-1) & mask) > 0).to(torch.uint8).view(packed.shape[0], -1)
    def pack_to_int8(self):
        if self.scale.dtype == torch.float16:
            minmin = 1e-4
            maxmax = 1e4
        else:
            minmin = 1e-6
            maxmax = 1e6
        scale = self.clamp_ste(self.scale, minmin, maxmax)
        weight = torch.clamp(self.round_ste(self.weight/scale+self.maxq/2), 0, self.maxq)
        del self.weight
        weight = weight.reshape(-1,  8).to(torch.uint8)
        shifts = (1 << torch.arange(8, dtype=torch.uint8, device=weight.device))
        packed_weight = (weight * shifts).sum(dim=-1).to(torch.uint8)
        self.register_buffer('packed_weight', packed_weight)
        self.packed_flag = True

    def forward(self, x):
        weight_fp = getattr(self, '_weight_fp', None)
        if weight_fp is not None:
            return F.linear(x, weight_fp.to(x), self.bias)
        weight = checkpoint(self.get_weight, use_reentrant=False)
        return F.linear(x, weight[:, :self.ic].to(x), self.bias)

    @torch.no_grad()
    def materialize(self, dtype=None):
        """Pre-dequantize the packed weight once and cache it as a buffer.

        ``get_weight()`` is a pure function (packed bits -> scale -> rotation ->
        reshape -> a1), so caching its output is bit-identical to recomputing it
        on every forward.  This removes the per-token dequantization cost during
        evaluation (especially for MoE, where top-k experts are re-dequantized
        per token).  The quantization-only storage (``packed_weight``/``scale``)
        is released to keep peak memory near the FP baseline.

        ``dtype`` casts the cached weight to the evaluation dtype **once**.
        Without it the raw ``get_weight()`` output stays float32
        (``unpack_bits_uint8`` returns uint8 and ``uint8 - 0.5`` promotes to
        float32 regardless of the module dtype), which made ``forward()``
        re-cast the whole weight matrix on every call (an O(weight) cost paid
        per routed MoE expert per forward) and doubled the resident weight
        memory.  The dequantization math itself is untouched: it still runs in
        float32 and is only cast at the end, exactly like the plain-``nn.Linear``
        path that loads a ``*.dequant.pth`` cache.
        """
        if hasattr(self, '_weight_fp'):
            return self
        if dtype is None:
            # After ``model.to(target_dtype)`` these already hold the evaluation
            # dtype, so the cached weight matches the activations.
            if 'scale' in self._parameters:
                dtype = self._parameters['scale'].dtype
            elif 'weight' in self._parameters:
                dtype = self._parameters['weight'].dtype
            else:
                dtype = torch.float32
        weight_fp = self.get_weight()[:, :self.ic]
        if weight_fp.dtype != dtype:
            weight_fp = weight_fp.to(dtype)
        self.register_buffer('_weight_fp', weight_fp.contiguous(), persistent=False)
        for name in ('packed_weight',):
            if name in self._buffers:
                del self._buffers[name]
        if 'scale' in self._parameters:
            del self._parameters['scale']
        return self

class TmpLinear(nn.Module):
    def __init__(
        self,
        org_module: nn.Linear,
        w_bits,
        expc = '32to16',
        training_trans = False, 
        groupsize = -1, 
        fast_nearest = True,
        shared = None
    ):
        super(TmpLinear, self).__init__()
        self.orilinear = org_module
        self.ic = self.orilinear.weight.data.shape[1]
        self.oc = self.orilinear.weight.data.shape[0]
        
        parts = expc.split('to')
        self.root = int(parts[1])
        if self.root == 8:
            if self.ic >10000:
                self.transdim2 = 128
            else:
                self.transdim2 = 64
        #self.transdim2 = round(math.sqrt(self.ic)/self.root)*self.root
        self.transdim1 = math.ceil(self.ic/self.transdim2)
        self.expic = self.transdim1 * self.transdim2
        
        self.orilinear.weight.data = F.pad(self.orilinear.weight.data, (0, self.expic - self.ic), mode="constant", value=0)
        if shared is not None:
            object.__setattr__(self, 'a1', shared.a1)
            object.__setattr__(self, 'a2', shared.a2)
            object.__setattr__(self, 'Trans', shared.Trans)
            object.__setattr__(self, 'shared', shared)
        else:
            self.a1 = nn.Parameter(torch.ones(self.expic).to(self.orilinear.weight))
            self.a2 = nn.Parameter(torch.ones(self.transdim1, self.transdim2//self.root).to(self.orilinear.weight))
            self.Trans = None
        #self.a3 = nn.Parameter(torch.ones(self.transdim1, self.transdim2//self.root).to(self.orilinear.weight))
        self.fwd_func = F.linear
        self.quantizer = Quantizer()
        self.quantizer.configure(
            w_bits, perchannel=True, sym=True, mse=True
        )

        self.input_trans = False
        self.output_trans = False
       
        self.rotation = None
        if self.orilinear.bias is not None:
            self.bias = self.orilinear.bias
        else:
            self.bias = None
        self.showflag=False

        self.norm = 2
        self.GPTQ = False
        self.training_trans = training_trans
        if shared is None and self.training_trans:
            if groupsize ==128 :
                #self.Trans = GHalfSVDDecomposeTransMatrix(self.transdim1//2 ,8,20)
                #self.Trans = HalfSVDDecomposeTransMatrix(self.transdim1, self.transdim2, diag_init = True)
                self.Trans = HalfSVDDecomposeTransMatrix(self.transdim1, self.transdim2)
            else:
                self.Trans = HalfSVDDecomposeTransMatrix(self.transdim1, self.transdim2)
        self.groupsize = groupsize

    def round_ste(self, x: torch.Tensor):
        """
        Implement Straight-Through Estimator for rounding operation.
        """
        return (x.round() - x).detach() + x
    def check_nan(self,x,name):
        non_finite_mask = ~torch.isfinite(x)
        indices = torch.nonzero(non_finite_mask)
        if indices.numel() > 0:
            print(f"find nan in {name}")
            print(x.shape)
            #for index in indices:
            #    print(f" {name}- 索引: {index.tolist()}, 值为: {x[tuple(index)]}")
        
    def quantize(self, x, scale, zero, maxq):
        
        if maxq < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        scale = torch.clamp(scale, 1e-6, 1e6)
        q = torch.clamp(self.round_ste(x / scale + zero) , 0, maxq)
        q = scale * (q - zero)
        
        
        return q
    
    def clamp_ste(self, x: torch.Tensor, min, max):
        return (x.clamp(min,max) - x).detach() + x

    def _a1(self):
        shared = getattr(self, 'shared', None)
        return shared.a1 if shared is not None else self.a1

    def _a2(self):
        shared = getattr(self, 'shared', None)
        return shared.a2 if shared is not None else self.a2

    def _trans(self):
        shared = getattr(self, 'shared', None)
        return shared.Trans if shared is not None else self.Trans

    def find_params(self):
        a1 = self._a1()
        if a1.data.shape != (self.transdim1, self.expic//self.transdim1):
            a1.data = a1.data.reshape(self.transdim1, self.expic//self.transdim1)
        self.weight = self.orilinear.weight.reshape(self.oc, self.transdim1, -1) * self.clamp_ste(a1,0.02,50)
        self.weight = self.weight.detach()
        if self.input_trans:
            if self.training_trans:
                self.weight = self._trans()(self.weight)
            else:
                self.weight = Hadamard_trans(self.weight, self.transdim1, self.transdim2)
        
        self.weight = self.weight * self._a2().repeat_interleave(self.root, dim=-1)
        
        if self.groupsize == -1:
            self.quantizer.find_params(self.weight.reshape(self.oc, -1), weight=True)
        elif self.groupsize == 128:
            self.quantizer.find_params(self.weight.reshape(-1, 160), weight=True)
       
    def quant_tmpweight(self):
        #self.check_nan(self.orilinear.weight,'oriweight')
        #print(self.a1.max(),self.a1.min())
        self.weight = self.orilinear.weight.reshape(self.oc, self.transdim1, -1)  * self.clamp_ste(self._a1(),0.02,50)
        #self.check_nan(self.weight,'transweight1')
        if self.input_trans:
            if self.training_trans:
                self.weight = self._trans()(self.weight)
            else:
                self.weight = Hadamard_trans(self.weight, self.transdim1, self.transdim2)
        #self.check_nan(self.weight,'transweight2')
        self.weight = self.weight * self._a2().repeat_interleave(self.root, dim=-1)
        #self.check_nan(self.weight,'transweight3')
        if self.showflag:
            print('tmp')
        
        if self.groupsize == -1:
            self.weight = self.weight.reshape(self.oc, -1)
        elif self.groupsize == 128:
            self.weight = self.weight.reshape(-1, 128)

        self.weight = self.quantize(self.weight, self.quantizer.scale*2*F.sigmoid(self.quantizer.alpha), self.quantizer.zero, self.quantizer.maxq)
            

        self.weight = self.weight.reshape(self.oc, self.transdim1, -1)
        
        #self.weight = self.weight * (self.a3 / self.a2).repeat_interleave(self.root, dim=-1)
        self.weight = self.weight / (self._a2()).repeat_interleave(self.root, dim=-1)
        if self.showflag:
            print('scale, a2',self.weight.flatten()[:16])
        if self.input_trans:
            if self.training_trans:
                self.weight = self._trans()(self.weight, inv_t=True)
            else:
                self.weight = Hadamard_trans(self.weight, self.transdim1, self.transdim2, inv=True)
        if self.showflag:
            print('hadmard',self.weight.flatten()[:16])
        self.weight = self.weight[:, :, :self.expic//self.transdim1] / self.clamp_ste(self._a1(),0.02,50)
        self.weight = self.weight.reshape(self.oc, self.expic)
        if self.showflag:
            print('clip a1',self.weight.flatten()[:16])

    def forward(self, input: torch.Tensor):
        #self.check_nan(input,'input')
        input = self.fwd_func(input, self.weight[:, :self.ic], self.bias) 
        #self.check_nan(input,'output')
        return input



def replace_TmpLinaer_with_FWTLinear(model, args, layers):
    for n,m in model.named_children():
        if isinstance(m, TmpLinear):
            for layer in layers:
                if layer in n:
                    print(n)
                    fwtl = FWTLinear()
                    fwtl.convert_form_tmplinear(m, bits=args.wbits, expc = args.expc, training_trans = args.training_trans, groupsize = args.groupsize, fast_nearest = args.fast_nearest)
                    fwtl.bit_channel_convert()
                    setattr(model, n, fwtl)
        else:
            replace_TmpLinaer_with_FWTLinear(m, args, layers)

def replace_TmpLinaer_with_FWTLinear_mix(model, args, layers, expc_list):
    for n,m in model.named_children():
        if isinstance(m, TmpLinear):
            for layer in layers:
                if layer in n:
                    print(n)
                    if 'q_proj' in n or'k_proj' in n or'v_proj' in n :  
                        expc = expc_list[0]
                    if 'o_proj' in n :  
                        expc = expc_list[1]
                    if 'up_proj' in n or 'gate_proj' in n :  
                        expc = expc_list[2]
                    if 'down_proj' in n :  
                        expc = expc_list[3]
                    fwtl = FWTLinear()
                    fwtl.convert_form_tmplinear(m, bits=args.wbits, expc = expc, training_trans = args.training_trans, groupsize = args.groupsize, fast_nearest = args.fast_nearest)
                    fwtl.bit_channel_convert()
                    setattr(model, n, fwtl)
        else:
            replace_TmpLinaer_with_FWTLinear_mix(m, args, layers, expc_list)


def replace_linear_with_TmpLinear(model, args):
    for n,m in model.named_children():
        if isinstance(m, nn.Linear):
            if 'q_proj' in n or'k_proj' in n or'v_proj' in n or'o_proj' in n or'in_proj_qkv' in n or 'in_proj_z' in n or'out_proj' in n or 'up_proj' in n or 'gate_proj' in n or 'down_proj' in n:
                if 'orilinear' not in n:
                    print(n)
                    setattr(model, n, TmpLinear(m, args.wbits, expc = args.expc, training_trans = args.training_trans, groupsize = args.groupsize, fast_nearest = args.fast_nearest))
        else:
            replace_linear_with_TmpLinear(m, args)


def replace_linear_with_TmpLinear_part(model, args, layers):
    for n,m in model.named_children():
        if isinstance(m, nn.Linear):
            for layer in layers:
                if layer in n:
                    if 'orilinear' not in n:
                        setattr(model, n, TmpLinear(m, 3, expc = args.expc, training_trans = args.training_trans, groupsize = args.groupsize, fast_nearest = args.fast_nearest))
        else:
            replace_linear_with_TmpLinear_part(m, args,layers)


def replace_linear_with_TmpLinear_mix(model, args, expc_list):
    for n,m in model.named_children():
        if isinstance(m, nn.Linear):
            if 'q_proj' in n or'k_proj' in n or'v_proj' in n or'o_proj' in n or 'up_proj' in n or 'gate_proj' in n or 'down_proj' in n:
                if 'orilinear' not in n:
                    if 'q_proj' in n or'k_proj' in n or'v_proj' in n :  
                        expc = expc_list[0]
                    if 'o_proj' in n :  
                        expc = expc_list[1]
                    if 'up_proj' in n or 'gate_proj' in n :  
                        expc = expc_list[2]
                    if 'down_proj' in n :  
                        expc = expc_list[3]
                    setattr(model, n, TmpLinear(m, args.wbits, expc = expc, training_trans = args.training_trans, groupsize = args.groupsize, fast_nearest = args.fast_nearest))
        else:
            replace_linear_with_TmpLinear_mix(m, args, expc_list)

def strtrans(inp):
    if inp =='n':
        return 2
    if inp =='m':
        return 2.3
    if inp =='h':
        return 2.7
    if inp =='p':
        return 3
    if inp =='nt':
        return 1.625
    if inp =='nm':
        return 2.071
    if inp =='nl':
        return 1.875
    if inp =='nh':
        return 2.78
    if inp =='np':
        return 2.25

_LATTICE_SEARCH_CACHE = {}

# Chunk size used by find_nearest_fast when batching multiple MoE experts.
# The nearest-neighbour search is per-row, so this only affects throughput and
# memory (not numerical results). Bump to 512+ if the GPU has headroom.
_NEAREST_BATCH_SIZE = 256


def get_lattice_search_cache(expc, device):
    """Build (once per expc/device) and cache the lattice search structure.

    Precomputes the lattice tensor ``T``, its null-space completion/inverse and
    the ``2**(root2-root)`` sign-candidate table that ``find_nearest_fast``
    needs. These only depend on ``expc``, so sharing them across the 128*3
    expert projections avoids recomputing them (and re-``torch.load``-ing the
    lattice file) hundreds of times.
    """
    key = (expc, str(device))
    cache = _LATTICE_SEARCH_CACHE.get(key)
    if cache is not None:
        return cache

    T = torch.load(f'./lattice/{expc}.pt').to(device).to(torch.float32)
    root = int(expc.split('to')[1])
    root2 = int(expc.split('to')[0])
    padding_length = root2 - root

    _, _, Vh = torch.linalg.svd(T)
    N = Vh[T.shape[0]:]
    M_square = torch.cat([T, N], dim=0)
    M_square_inv = torch.linalg.inv(M_square)
    padding_vectors = torch.tensor(
        list(product([-1, 1], repeat=padding_length)),
        dtype=torch.float32, device=device,
    )

    cache = {
        'T': T,
        'M_square_inv': M_square_inv,
        'padding_vectors': padding_vectors,
        'num_candidates': padding_vectors.shape[0],
    }
    _LATTICE_SEARCH_CACHE[key] = cache
    return cache


def _is_moe_block(module):
    return (
        hasattr(module, "experts")
        and isinstance(module.experts, nn.ModuleList)
        and len(module.experts) > 0
        and hasattr(module.experts[0], "gate_proj")
    )


def _replace_moe_block_grouped(moe_block, args, num_groups):
    num_experts = len(moe_block.experts)
    group_size = math.ceil(num_experts / num_groups)
    proj_types = ["gate_proj", "up_proj", "down_proj"]

    moe_block.moe_shared = nn.ModuleList()
    moe_block.moe_num_groups = num_groups
    moe_block.moe_group_size = group_size

    for g in range(num_groups):
        holders = nn.ModuleDict()
        for p in proj_types:
            ic = getattr(moe_block.experts[g * group_size], p).in_features
            holders[p] = MoeSharedRotScale(ic, args.expc, args.training_trans, args.groupsize)
        moe_block.moe_shared.append(holders)

        for e in range(g * group_size, min((g + 1) * group_size, num_experts)):
            expert = moe_block.experts[e]
            for p in proj_types:
                lin = getattr(expert, p)
                setattr(
                    expert,
                    p,
                    TmpLinear(
                        lin,
                        args.wbits,
                        expc=args.expc,
                        training_trans=args.training_trans,
                        groupsize=args.groupsize,
                        fast_nearest=args.fast_nearest,
                        shared=holders[p],
                    ),
                )


def replace_linear_with_TmpLinear_moe(model, args, num_groups=1):
    for n, m in model.named_children():
        if _is_moe_block(m):
            _replace_moe_block_grouped(m, args, num_groups)
        elif isinstance(m, nn.Linear):
            if (
                'q_proj' in n or 'k_proj' in n or 'v_proj' in n or 'o_proj' in n
                or 'in_proj_qkv' in n or 'in_proj_z' in n or 'out_proj' in n
            ):
                if 'orilinear' not in n:
                    print(n)
                    setattr(
                        model,
                        n,
                        TmpLinear(
                            m,
                            args.wbits,
                            expc=args.expc,
                            training_trans=args.training_trans,
                            groupsize=args.groupsize,
                            fast_nearest=args.fast_nearest,
                        ),
                    )
        else:
            replace_linear_with_TmpLinear_moe(m, args, num_groups)


def prepare_moe_shared_for_export(holder, T, device):
    root = holder.root
    root2 = holder.root2
    transdim2 = holder.transdim2

    holder.a2.data = (1.0 / holder.a2.data).reshape(-1).repeat_interleave(root2)

    if holder.Trans is not None:
        M = torch.zeros(transdim2 // root * root2, transdim2, device=device).to(T)
        for i in range(transdim2 // root):
            M[i * root2:(i + 1) * root2, i * root:(i + 1) * root] = T.t()
        mr = holder.Trans.linear_right.data.to(device)
        holder.Trans.linear_right.data = (M @ mr).to(holder.Trans.linear_right)


def _convert_moe_block_grouped(moe_block, args, layers, structure_only, search_rank=None, search_world_size=None):
    proj_types = ["gate_proj", "up_proj", "down_proj"]
    active = [p for p in proj_types if any(layer in p for layer in layers)]
    if not active:
        return

    num_experts = len(moe_block.experts)
    num_groups = len(moe_block.moe_shared)
    group_size = getattr(moe_block, "moe_group_size", math.ceil(num_experts / num_groups))
    sharded = (search_world_size is not None and search_world_size > 1)

    for g in range(num_groups):
        holders = moe_block.moe_shared[g]
        start = g * group_size
        end = min((g + 1) * group_size, num_experts)

        for p in active:
            holder = holders[p]
            expc = args.expc

            # Step 1: convert each expert TmpLinear -> FWTLinear (shared refs, no buffer yet)
            for e in range(start, end):
                tmp = getattr(moe_block.experts[e], p)
                if not isinstance(tmp, TmpLinear):
                    continue
                fwt = FWTLinear()
                fwt.convert_form_tmplinear(
                    tmp,
                    bits=args.wbits,
                    expc=expc,
                    training_trans=args.training_trans,
                    groupsize=args.groupsize,
                    fast_nearest=args.fast_nearest,
                    shared=holder,
                    to_buffer=False,
                )
                setattr(moe_block.experts[e], p, fwt)

            # Step 2: group-shared materialization once per (group, projection)
            if holder.Trans is not None:
                holder.Trans.to_buffer()
            fwt0 = getattr(moe_block.experts[start], p)
            cache = get_lattice_search_cache(expc, fwt0.weight.device)
            prepare_moe_shared_for_export(holder, cache['T'], fwt0.weight.device)

            # Step 3: per-expert vector quantization. When sharding, each rank
            # performs the real search only for the experts it owns
            # (e % world_size == rank); the rest are left as placeholder null
            # weights and later filled in via all_gather. The search is per-row,
            # so sharding is bit-identical to a single-rank full search.
            fwts = [getattr(moe_block.experts[e], p) for e in range(start, end)]
            if structure_only:
                for fwt in fwts:
                    fwt.bit_channel_convert_shared(fast=True, cache=cache)
                continue

            if sharded:
                owned = [e for e in range(start, end) if e % search_world_size == search_rank]
                for e in range(start, end):
                    if e in owned:
                        continue
                    fwt = getattr(moe_block.experts[e], p)
                    fwt.bit_channel_convert_shared(fast=True, cache=cache)
                if not owned:
                    continue
                owned_fwts = [getattr(moe_block.experts[e], p) for e in owned]
            else:
                owned_fwts = fwts

            if all(getattr(fwt, 'fast_nearest', False) for fwt in owned_fwts):
                # Prepare owned experts, run a single batched nearest-neighbour
                # search, then finalize per expert. The search is per-row, so
                # concatenating experts is bit-identical to per-expert calls.
                l2s = []
                norm_weights = []
                for fwt in owned_fwts:
                    l2, nw = fwt.prepare_shared_search()
                    l2s.append(l2)
                    norm_weights.append(nw)
                nw_all = torch.cat(norm_weights, dim=0)
                q_all = owned_fwts[0].find_nearest_fast(
                    nw_all,
                    cache['T'],
                    owned_fwts[0].root2 - owned_fwts[0].root,
                    batch_size=_NEAREST_BATCH_SIZE,
                    device=owned_fwts[0].weight.device,
                    cache=cache,
                )
                qs = q_all.split([nw.shape[0] for nw in norm_weights], dim=0)
                for fwt, q, l2 in zip(owned_fwts, qs, l2s):
                    fwt.finalize_shared_search(q, l2)
            else:
                # Exact (slow) search path — keep per-expert to preserve the
                # original find_nearest_in_batches behaviour.
                for fwt in owned_fwts:
                    fwt.bit_channel_convert_shared(fast=False, cache=cache)


def replace_TmpLinaer_with_FWTLinear_moe(model, args, layers, expc_list=None, structure_only=False, search_rank=None, search_world_size=None):
    for n, m in model.named_children():
        if _is_moe_block(m):
            _convert_moe_block_grouped(m, args, layers, structure_only, search_rank, search_world_size)
        elif isinstance(m, TmpLinear):
            for layer in layers:
                if layer in n:
                    print(n)
                    expc = args.expc
                    if expc_list is not None:
                        if 'q_proj' in n or 'k_proj' in n or 'v_proj' in n:
                            expc = expc_list[0]
                        elif 'o_proj' in n:
                            expc = expc_list[1]
                        elif 'up_proj' in n or 'gate_proj' in n:
                            expc = expc_list[2]
                        elif 'down_proj' in n:
                            expc = expc_list[3]
                    fwt = FWTLinear()
                    fwt.convert_form_tmplinear(
                        m,
                        bits=args.wbits,
                        expc=expc,
                        training_trans=args.training_trans,
                        groupsize=args.groupsize,
                        fast_nearest=args.fast_nearest,
                    )
                    if structure_only:
                        fwt.bit_channel_convert(fast=True)
                    else:
                        fwt.bit_channel_convert()
                    setattr(model, n, fwt)
        else:
            replace_TmpLinaer_with_FWTLinear_moe(m, args, layers, expc_list, structure_only, search_rank, search_world_size)


def make_fwtlinear_eval(lin, wbits, expc, device='cpu', dtype=torch.float16):
    """Build a single dense FWTLinear eval structure with null weights.

    Mirrors the structure-building in ``e2e_utils.load_quantized_model`` so the
    resulting module has the same keys/shapes as a packed LiftQuant checkpoint.
    """
    fake = nn.Linear(lin.in_features, lin.out_features, lin.bias is not None, device=device, dtype=dtype)
    tmp = TmpLinear(fake, wbits, expc=expc, training_trans=True, groupsize=-1)
    tmp.find_params()
    tmp.quantizer.alpha = nn.Parameter(torch.zeros(tmp.quantizer.scale.shape, device=device, dtype=dtype))
    fwt = FWTLinear()
    fwt.convert_form_tmplinear(tmp, bits=wbits, expc=expc, training_trans=True, groupsize=-1)
    fwt.bit_channel_convert(True)
    fwt.pack_to_int8()
    return fwt


def build_moe_grouped_fwt_eval(moe_block, wbits, expc, num_groups, device='cpu', dtype=torch.float16):
    """Build the grouped FWTLinear eval structure for a MoE block.

    Replaces each expert ``gate_proj/up_proj/down_proj`` ``nn.Linear`` with an
    ``FWTLinear`` that references group-shared ``MoeSharedRotScale`` holders, so
    the structure matches a grouped LiftQuant checkpoint
    (``mlp.moe_shared.{g}.{proj}.*`` + per-expert ``scale/packed_weight``).
    """
    num_experts = len(moe_block.experts)
    group_size = math.ceil(num_experts / num_groups)
    proj_types = ['gate_proj', 'up_proj', 'down_proj']

    moe_block.moe_shared = nn.ModuleList()
    moe_block.moe_num_groups = num_groups
    moe_block.moe_group_size = group_size

    for g in range(num_groups):
        holders = nn.ModuleDict()
        for p in proj_types:
            ic = getattr(moe_block.experts[g * group_size], p).in_features
            holder = MoeSharedRotScale(ic, expc, training_trans=True, groupsize=-1).to(device=device)
            # Keep the orthogonal rotation in float32 (the cayley parametrization
            # needs LU solve), while scaling params match the checkpoint dtype.
            holder.a1.data = holder.a1.data.to(dtype=dtype)
            holder.a2.data = holder.a2.data.to(dtype=dtype)
            holders[p] = holder
        moe_block.moe_shared.append(holders)

        start = g * group_size
        end = min((g + 1) * group_size, num_experts)
        for p in proj_types:
            holder = holders[p]
            # Step 1: convert each expert Linear -> FWTLinear (shared refs, no buffer yet)
            for e in range(start, end):
                lin = getattr(moe_block.experts[e], p)
                fake = nn.Linear(lin.in_features, lin.out_features, lin.bias is not None, device=device, dtype=dtype)
                tmp = TmpLinear(fake, wbits, expc=expc, training_trans=True, groupsize=-1, shared=holder)
                tmp.find_params()
                tmp.quantizer.alpha = nn.Parameter(torch.zeros(tmp.quantizer.scale.shape, device=device, dtype=dtype))
                fwt = FWTLinear()
                fwt.convert_form_tmplinear(
                    tmp,
                    bits=wbits,
                    expc=expc,
                    training_trans=True,
                    groupsize=-1,
                    shared=holder,
                    to_buffer=False,
                )
                setattr(moe_block.experts[e], p, fwt)
            # Step 2: group-shared materialization once per (group, projection)
            if holder.Trans is not None:
                holder.Trans.to_buffer()
            T = torch.load('./lattice/' + expc + '.pt').to(device)
            prepare_moe_shared_for_export(holder, T, device)
            # Step 3: per-expert fast vector quantization + packing
            for e in range(start, end):
                fwt = getattr(moe_block.experts[e], p)
                fwt.bit_channel_convert_shared(fast=True)
                fwt.pack_to_int8()


def get_layer_parameters(model, bitslist):
    pnums = 0
    lbytes = 0
    tmpnums = 0
    for n,m in model.named_modules():
        if isinstance(m, nn.Linear):
            if 'q_proj' in n or'k_proj' in n or'v_proj' in n :
                tmpnums += m.weight.shape[0]*m.weight.shape[1]
    lbytes += tmpnums * strtrans(bitslist[0])
    pnums += tmpnums

    tmpnums = 0
    for n,m in model.named_modules():
        if isinstance(m, nn.Linear):
            if 'o_proj' in n :
                tmpnums += m.weight.shape[0]*m.weight.shape[1]
    lbytes += tmpnums * strtrans(bitslist[1])
    pnums += tmpnums

    tmpnums = 0
    for n,m in model.named_modules():
        if isinstance(m, nn.Linear):
            if 'up_proj' in n or'gate_proj' in n  :
                tmpnums += m.weight.shape[0]*m.weight.shape[1]
    lbytes += tmpnums * strtrans(bitslist[2])
    pnums += tmpnums

    tmpnums = 0
    for n,m in model.named_modules():
        if isinstance(m, nn.Linear):
            if 'down_proj' in n :
                tmpnums += m.weight.shape[0]*m.weight.shape[1]
    lbytes += tmpnums * strtrans(bitslist[3])
    pnums += tmpnums
    return pnums, lbytes