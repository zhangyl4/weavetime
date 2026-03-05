import math
import os
import torch
from typing import Optional, Tuple, List, Dict, Any
from transformers.cache_utils import Cache, DynamicCache

from .dot_production_attention import get_multi_stage_dot_production_attention


# Allocate a fixed-size block of GPU memory specifically for storing the KV-Cache of the local_window.
class CudaCache:
    def __init__(self, num_units, unit_size, dtype):
        self.num_units = num_units  # n_block
        self.unit_size = unit_size  # block_size * hidden_dim * 2
        self.dtype = dtype
        self.data = torch.empty(
            (num_units, unit_size),
            device = "cuda",
            dtype=dtype
        )
        self.idle_set = set(list(range(num_units)))

    def alloc(self):
        assert len(self.idle_set) > 0
        idx = self.idle_set.pop()
        return self.data[idx], idx

    def delete(self, idx):
        assert idx not in self.idle_set
        self.idle_set.add(idx)


# The KV-Cache management unit supports data transfer between the CPU and GPU.
class MemoryUnit:
    # Initialize the KV-Cache management unit and store it on the CPU.
    def __init__(
        self, 
        kv: Tuple[torch.Tensor, torch.Tensor], 
        cache: CudaCache, 
        load_to_cache: bool = False, 
        pin_memory: bool = False,
    ):
        self.cache = cache

        if kv[0].is_cuda:
            cpu_data = tuple(_t.contiguous().to("cpu", non_blocking=True) for _t in kv)
        else:
            cpu_data = tuple(_t.contiguous() for _t in kv)

        if pin_memory:
            cpu_data = tuple(_t.pin_memory() for _t in cpu_data)

        if load_to_cache:
            gpu_data, gpu_data_id = cache.alloc()
            gpu_data = gpu_data.view((2,) + kv[0].shape)
            gpu_data[0].copy_(kv[0], non_blocking=True)
            gpu_data[1].copy_(kv[1], non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            gpu_data, gpu_data_id = None, None
            event = None

        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id
        self.event = event

    # Load data from the CPU to the GPU and copy it to 'target' when necessary.
    # target: 2x (n_head, n_token, head_dim), on GPU
    def load(self, target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> bool:
        if self.gpu_data is not None:
            if target is not None:
                target[0].copy_(self.gpu_data[0], non_blocking=True)
                target[1].copy_(self.gpu_data[1], non_blocking=True)
                target_event = torch.cuda.Event()
                target_event.record(torch.cuda.current_stream())
            else:
                target_event = None

            return False, target_event

        gpu_data, gpu_data_id = self.cache.alloc()
        gpu_data = gpu_data.view((2,) + self.cpu_data[0].shape)
        if target is not None:
            target[0].copy_(self.cpu_data[0], non_blocking=True)
            target[1].copy_(self.cpu_data[1], non_blocking=True)
            target_event = torch.cuda.Event()
            target_event.record(torch.cuda.current_stream())
            gpu_data[0].copy_(target[0], non_blocking=True)
            gpu_data[1].copy_(target[1], non_blocking=True)

        else:
            gpu_data[0].copy_(self.cpu_data[0], non_blocking=True)
            gpu_data[1].copy_(self.cpu_data[1], non_blocking=True)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.event = event
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id

        return True, target_event

    # Get the KV-Cache stored on GPU
    def get(self):
        assert self.gpu_data is not None
        self.event.wait()
        return self.gpu_data

    # Clear the KV-Cache stored on GPU
    def offload(self):
        assert self.gpu_data is not None
        self.event.wait()
        self.gpu_data = None
        self.cache.delete(self.gpu_data_id)
        self.gpu_data_id = None

    def calculate_cpu_memory(self):
        return len(self.cpu_data) * self.cpu_data[0].numel() * self.cpu_data[0].element_size()


# A dynamically growing vector cache on the GPU, used to store representative vectors of video frames.
class VectorTensor:
    # Initialize an empty cache of size (16, hidden_dim) on the GPU.
    def __init__(
        self, 
        hidden_size,
        element_dtype,
        device
    ):
        init_cached_size = 16
        self.data = torch.empty(
            (init_cached_size, hidden_size),
            dtype=element_dtype,
            device=device
        )
        self.length = 0
        self.cache_size = init_cached_size
        self.hidden_size = hidden_size

    # Double the size of the cache.
    def append_cache(self):
        new_cache_size = self.cache_size * 2
        data_shape = self.data.shape
        new_data = torch.empty(
            (new_cache_size,) + data_shape[1:],
            device=self.data.device,
            dtype=self.data.dtype
        )
        new_data[:self.cache_size,...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    # Append a frame vector to the cache, and expand the cache if it exceeds the current cache size.
    def append(self, tensor: torch.Tensor):
        assert tensor.dtype == self.data.dtype
        assert tensor.size(1) == self.hidden_size, f'{tensor.size(1)}, {self.hidden_size}'
        assert tensor.is_contiguous()

        append_l = tensor.size(0)

        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length: self.length+append_l, ...].copy_(tensor)

        self.length += append_l

    # Get the cached frame vectors
    def get_data(self):
        return self.data[:self.length, ...]

    def get_cosine_similarity(self, tensor: torch.Tensor, num_heads: int = None, dim_head: int = None, head_weights: torch.Tensor = None):
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size, f'{tensor.size(0)}, {self.hidden_size}'
        key = self.data[:self.length].float()  # (T, D), convert to fp32 to prevent numerical overflow
        query = tensor[None, :].float()  # (1, D)

        logits = torch.matmul(query, key.T)[0]  # (T,)

        assert logits.dim() == 1 and logits.size(0) == self.length
        return logits

    def get_maxsim_similarity_chunked(self, query_tokens: torch.Tensor, block_size: int, chunk_frames: int = 8):
        """
        Memory-efficient maxSim similarity calculation using chunked processing.
        
        Args:
            query_tokens: (seq_len, hidden_size) query token representations
            block_size: number of tokens per frame
            chunk_frames: number of frames to process at once to save memory
            
        Returns:
            logits: (num_frames,) similarity scores for each frame
        """
        assert query_tokens.dim() == 2 and query_tokens.size(1) == self.hidden_size
        
        query_tokens = query_tokens.float()  # (seq_len, D)
        frame_tokens = self.data[:self.length].float()  # (T, D)
        
        # Reshape frame tokens into blocks (frames)
        num_frames = self.length // block_size
        if num_frames == 0:
            return torch.zeros(0, device=query_tokens.device)
        
        # Only use complete frames
        used_tokens = num_frames * block_size
        frame_tokens = frame_tokens[:used_tokens].view(num_frames, block_size, self.hidden_size)  # (num_frames, block_size, D)
        
        frame_scores = []
        
        # Process frames in chunks to reduce memory usage
        for start_frame in range(0, num_frames, chunk_frames):
            end_frame = min(start_frame + chunk_frames, num_frames)
            chunk_frame_tokens = frame_tokens[start_frame:end_frame]  # (chunk_size, block_size, D)
            
            # Calculate similarity for this chunk
            # query_tokens: (seq_len, D), chunk_frame_tokens: (chunk_size, block_size, D)

            # calculate similarity: (seq_len, D) x (chunk_size, block_size, D) -> (seq_len, chunk_size, block_size)
            similarity = torch.einsum('ld,fbd->lfb', query_tokens, chunk_frame_tokens)
            
            # For each query token, find the maximum similarity with any token in each frame
            max_sim_per_query = similarity.max(dim=-1)[0]  # (seq_len, chunk_size)
            
            # For each frame, take the maximum similarity across all query tokens
            chunk_frame_scores = max_sim_per_query.max(dim=0)[0]  # (chunk_size,)
            
            frame_scores.append(chunk_frame_scores)
            
            # Clear intermediate tensors to free memory
            del similarity, max_sim_per_query, chunk_frame_scores
        
        return torch.cat(frame_scores, dim=0)

    def __len__(self):
        return self.length


GLOBAL_STREAM = None


class ContextManager:
    def __init__(self, 
                 position_embedding,
                 n_init, n_local, 
                 block_size, max_cached_block, topk, chunk_size, exc_block_size, 
                 fattn: bool = False,
                 async_global_stream: bool = False,
                 pin_memory: bool = False,
                 use_hybrid_similarity: bool = True,
    ):

        self.length = 0  # number of tokens in the KV-Cache
        self.position_embedding = position_embedding
        self.n_init = n_init
        self.n_local = n_local
        self.block_size = block_size
        self.max_cached_block = max_cached_block
        self.exc_block_size = exc_block_size
        assert exc_block_size <= n_local # no global token in input
        self.topk = topk
        self.chunk_size = chunk_size
        self.Attn, _ = get_multi_stage_dot_production_attention(fattn)
        self.fattn = fattn
        self.initialized = False
        self.load_count = 0
        self.async_global_stream = async_global_stream
        self.pin_memory = pin_memory
        self.use_hybrid_similarity = use_hybrid_similarity
        global GLOBAL_STREAM
        if self.async_global_stream and GLOBAL_STREAM is None:
            GLOBAL_STREAM = torch.cuda.Stream()

        self.reset_retrieval()
        
        # Query states saving for entropy-adaptive retrieval
        self.saved_query_states: Optional[torch.Tensor] = None
        self.save_query_states_flag: bool = False
        self.use_saved_query_states_flag: bool = False
        
        # Storage policy (applied when offloading frames into global_blocks)
        self.storage_mode = None  # None | 'rate' | 'similarity'
        self._storage_rate_step = 1
        self.storage_threshold = 0.99
        self._storage_rate_counter = []
        self._storage_last_avg = []
        # global decision state to keep alignment across units
        self._storage_rate_counter_global = 0
        self._storage_last_avg_global = None

    def _remove_lru_blocks(self, u, num_remove: Optional[int] = None, ignore_blocks = None):
        if num_remove is None:
            num_remove = len(self.cached_blocks[u]) - self.max_cached_block

        if num_remove <= 0:
            return

        lst = list(self.cached_blocks[u].items())
        lst.sort(key=lambda x: x[1])

        removed = 0
        for i in range(len(lst)):
            idx = lst[i][0]
            if ignore_blocks is None or (idx not in ignore_blocks):
                self.global_blocks[u][idx].offload()
                self.cached_blocks[u].pop(idx)
                removed += 1

            if removed >= num_remove:
                return

    # handle GQA, k: (batch_size, n_head_kv, length, dim_head) -> (batch_size, n_head, length, dim_head)
    def _from_group_kv(self, tensor):
        # tensor: (batch_size, n_head_kv, length, dim_head)
        assert tensor.dim() == 4 
        assert tensor.size(1) == self.num_heads_kv
        if self.num_heads == self.num_heads_kv:
            return tensor
        _, _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv
        tensor = tensor.view((self.num_units, self.unit_size_kv, 1, length, dim_head))  # (batch_size, n_head_kv, 1, length, dim_head)
        tensor = tensor.expand((self.num_units, self.unit_size_kv, num_group, length, dim_head)).reshape((self.num_units, self.num_heads, length, dim_head))  # (batch_size, n_head, length, dim_head)
        return tensor
    
    def init(
        self, 
        local_q, local_k, local_v,
        global_q, global_k, global_v
    ):
        """
        Only use the metadata of these parameters, such as shape, dtype, and device.
        """
        assert local_q.dim() == 4
        batch_size, num_heads, len_q, dim_head = local_q.shape
        num_heads_kv = local_k.size(1)

        for _t in [local_q, local_k, local_v, global_q, global_k, global_v]:
            assert _t.size(0) == batch_size
            assert (_t.size(1) == num_heads or _t.size(1) == num_heads_kv)
            assert _t.size(2) == len_q
            assert _t.size(3) == dim_head
            assert _t.is_cuda

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.num_units = batch_size
        self.unit_size = num_heads
        self.unit_size_kv = num_heads_kv
        
        # initialize storage policy state per unit
        self._storage_rate_counter = [0 for _ in range(self.num_units)]
        self._storage_last_avg = [None for _ in range(self.num_units)]

        self.global_blocks = [[] for _ in range(self.num_units)] # context memory's KV-Cache: [ batch_size x [memory_unit] ]
        self.cached_blocks = [{} for _ in range(self.num_units)] # relavency scores of blocks: batch_size x {block_id: block_score}
        self.num_global_block = 0

        # context memory's representative keys: batch_size x (n_blocks, hidden_dim)
        self.block_k = [VectorTensor(
            dim_head * self.unit_size, global_k.dtype, global_k.device
        ) for _ in range(self.num_units)]

        # local KV
        self.local_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_k.dtype, device=local_k.device)  # (batch_size, n_head_kv, 0, dim_head)
        self.local_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=local_v.dtype, device=local_v.device)

        # global KV that are not yet processed into blocks.
        # 2 x (batch_size, n_head_kv, length, dim_head)
        self.global_remainder = (
            torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device),
            torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_v.dtype, device=global_v.device),
        )

        # init KV
        self.init_k = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_v = torch.empty((self.num_units, self.unit_size_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_exc = False
        self.dtype = local_q.dtype
        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, local_k.device, local_k.dim()
        )

        # buffering global KV during attention computations
        # (2, batch_size, n_head_kv, L, dim_head)
        # L = n_init + n_retrieve
        buffer_len = self.topk * self.block_size + self.n_init
        self.global_buffer = torch.zeros(
                (2, self.num_units, self.unit_size_kv, buffer_len , dim_head),
                dtype = global_k.dtype, device=global_k.device
            )
        self.global_buffer_init_st = 0
        self.global_buffer_init_ed = 0
        self.cuda_cache = CudaCache(
            self.max_cached_block * self.num_units,
            self.unit_size_kv * self.block_size * dim_head * 2,
            local_k.dtype
        )  # (max_cached_block * batch_size, block_size * D * 2)

        self.initialized = True

    def set_retrieval(self):
        self.to_retrieve = True

    def reset_retrieval(self):
        self.similarity = None
        self.retrieved_block_indices = None
        self.to_retrieve = False

    def save_query_states(self, save: bool):
        """Set flag to save query states during forward pass."""
        self.save_query_states_flag = bool(save)

    def use_saved_query_states(self, use: bool):
        """Set flag to use saved query states for retrieval instead of new ones."""
        self.use_saved_query_states_flag = bool(use)

    def clear_saved_query_states(self):
        """Clear saved query states and reset flags."""
        self.saved_query_states = None
        self.save_query_states_flag = False
        self.use_saved_query_states_flag = False

    # Dynamically adjust block_size (tokens per frame) based on current video HxW.
    # This recreates block-dependent buffers and caches. Should be called before encoding video frames.
    def set_block_size(self, new_block_size: int):
        if not isinstance(new_block_size, int) or new_block_size <= 0:
            return
        if new_block_size == self.block_size:
            return
        self.block_size = new_block_size
        # Align processing block size with frame block size (bounded by n_local)
        self.exc_block_size = min(self.n_local, new_block_size)
        if not self.initialized:
            # Will allocate with the new block_size during the first append
            return
        # Recreate global buffer with new block size capacity
        buffer_len = self.topk * self.block_size + self.n_init
        dtype = self.global_remainder[0].dtype
        device = self.global_remainder[0].device
        self.global_buffer = torch.zeros(
            (2, self.num_units, self.unit_size_kv, buffer_len, self.dim_head),
            dtype=dtype,
            device=device,
        )
        self.global_buffer_init_st = 0
        self.global_buffer_init_ed = 0
        # Reset global blocks and indices because representation depends on block partitioning
        self.global_blocks = [[] for _ in range(self.num_units)]
        self.cached_blocks = [{} for _ in range(self.num_units)]
        self.num_global_block = 0
        # Reset representative keys storage
        self.block_k = [VectorTensor(
            self.dim_head * self.unit_size, dtype, device
        ) for _ in range(self.num_units)]
        # Recreate CUDA cache with new unit size
        self.cuda_cache = CudaCache(
            self.max_cached_block * self.num_units,
            self.unit_size_kv * self.block_size * self.dim_head * 2,
            dtype,
        )

    def set_retrieved_block_indices(self, retrieved_block_indices):
        # retrieved_block_indices (list): batch_size x n_frames
        if isinstance(retrieved_block_indices, torch.Tensor):
            retrieved_block_indices = retrieved_block_indices.cpu().tolist()
        self.retrieved_block_indices = retrieved_block_indices

    def get_retrieved_kv(self, query=None):
        """retrieve context blocks with retrieved_block_indices
        query: (batch_size, num_heads, length, dim_head)
        return [init_k, retrieved_k] and the respective v
        """
        # Use saved query_states if flag is set and saved query_states exists
        if self.use_saved_query_states_flag and self.saved_query_states is not None:
            query = self.saved_query_states

        if query is not None:  # retrieve based on the attention score between query and context's representative keys
            block_topk = self._calc_block_topk(query)
            self.set_retrieved_block_indices(block_topk)

        assert len(self.retrieved_block_indices) == self.num_units

        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        with torch.cuda.stream(GLOBAL_STREAM):
            if self.init_exc:  # init KV were loaded in global_h_k, context KV were offloaded in global_blocks
                # offload LRU blocks
                for u in range(self.num_units):
                    num_remove = len(self.cached_blocks[u]) - self.max_cached_block
                    for b_idx in self.retrieved_block_indices[u]:
                        if b_idx not in self.cached_blocks[u]:
                            num_remove += 1
                    self._remove_lru_blocks(u, num_remove, self.retrieved_block_indices[u])

                self.load_count += 1
                for u in range(self.num_units):
                    for b_idx in self.retrieved_block_indices[u]:
                        self.cached_blocks[u][b_idx] = self.load_count
                
                # no need to load init KV
                init_st = 0
                init_ed = init_st + self.init_k.size(-2)
                ed = init_ed
                assert self.global_buffer_init_st == init_st or self.global_buffer_init_ed == init_ed

                # load retrieved context KV
                for u in range(self.num_units):
                    # assert len(self.retrieved_block_indices[u]) == block_num
                    assert self.retrieved_block_indices[u][-1] < self.num_global_block, f'{self.retrieved_block_indices[u][-1]}, {self.num_global_block}'
                    for cnt, b_idx in enumerate(self.retrieved_block_indices[u]):
                        # load global_blocks[u][b_idx] onto GPU and make a copy to (global_h_k, global_h_v)
                        st = init_ed + cnt * self.block_size
                        ed = st + self.block_size
                        self.global_blocks[u][b_idx].load((global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :]))

            else:  # init KV and context are in self.global_remainder
                # load init KV
                init_st = 0
                init_ed = init_st + self.n_init
                global_h_k[:, :, init_st:init_ed] = self.global_remainder[0][:, :, init_st:init_ed]
                global_h_v[:, :, init_st:init_ed] = self.global_remainder[1][:, :, init_st:init_ed]
                ed = init_ed

                # load retrieved context KV
                for u in range(self.num_units):
                    # assert len(self.retrieved_block_indices[u]) == block_num
                    for cnt, b_idx in enumerate(self.retrieved_block_indices[u]):
                        remainder_st = init_ed + b_idx * self.block_size
                        remainder_ed = remainder_st + self.block_size
                        if remainder_st >= self.global_remainder[0].size(2):
                            break
                        st = init_ed + cnt * self.block_size
                        ed = st + self.block_size
                        global_h_k[u, :, st:ed] = self.global_remainder[0][u, :, remainder_st:remainder_ed]
                        global_h_v[u, :, st:ed] = self.global_remainder[1][u, :, remainder_st:remainder_ed]

            global_h_k = global_h_k[:, :, :ed, :]
            global_h_v = global_h_v[:, :, :ed, :]
            # assert global_h_k.size(-2) == global_h_v.size(-2) == self.n_init + block_num * self.block_size

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        assert global_h_k.size(-2) <= self.n_init + self.n_local
        return global_h_k, global_h_v 

    # Get the indices of the top-k vectors in self.block_k[u] that have the highest similarity with global_h_q[u].
    # ret: batch_size x topk
    def _calc_block_topk(
        self, global_h_q
    ):
        # Use all query tokens for maxSim instead of just the last one
        query_tokens = global_h_q  # (batch_size, num_heads, length, dim_head)
        assert query_tokens.shape[:2] == (self.num_units, self.unit_size)
        # Reshape to (batch_size, length, num_heads * dim_head) for maxSim calculation
        query_tokens = query_tokens.transpose(1, 2)  # (batch_size, length, num_heads, dim_head)
        query_tokens = query_tokens.reshape(self.num_units, query_tokens.size(1), -1)  # (batch_size, length, num_heads * dim_head)
        logits = None

        if self.num_global_block <= self.topk:
            if not self.init_exc:  # The local window has not yet been filled, i.e., KV-Cache offloading has not been activated. Retrieval needs to be performed within the local window.
                assert self.global_remainder[0].size(-2) > self.n_init, f'{self.global_remainder[0].shape}, {self.n_init}'
                global_k = self.global_remainder[0][:, :, self.n_init:, :]  # (batch_size, n_head_kv, length - n_init, dim_head)
                global_k = self._from_group_kv(global_k)  # (batch_size, num_heads, length - n_init, dim_head)

                assert global_k.size(-2) % self.block_size == 0, f'{global_k.shape}'
                block_num = global_k.size(-2) // self.block_size  # number of frames in local window
                if block_num <= self.topk:
                    ret = [list(range(block_num)) for _ in range(self.num_units)]
                else:
                    # Prepare tokens per frame
                    global_k = global_k.transpose(1, 2)  # (batch_size, length - n_init, num_heads, dim_head)
                    global_k = global_k.reshape(self.num_units, block_num, self.block_size, self.unit_size * self.dim_head)  # (batch_size, block_num, block_size, dim)

                    if self.use_hybrid_similarity:
                        # Memory-efficient chunked maxSim calculation for local window
                        logits_list = []
                        chunk_frames = 4  # Process fewer frames at once to save memory
                        
                        for u in range(self.num_units):
                            query_u = query_tokens[u]  # (length, D)
                            frames_u = global_k[u]  # (block_num, block_size, D)
                            frame_scores = []
                            
                            # Process frames in chunks
                            for start_frame in range(0, block_num, chunk_frames):
                                end_frame = min(start_frame + chunk_frames, block_num)
                                chunk_frames_u = frames_u[start_frame:end_frame]  # (chunk_size, block_size, D)
                                
                                # Calculate similarity for this chunk: (length, D) x (chunk_size, block_size, D) -> (length, chunk_size, block_size)
                                similarity = torch.einsum('ld,fbd->lfb', query_u.float(), chunk_frames_u.float())
                                max_sim_per_query = similarity.max(dim=-1)[0]  # (length, chunk_size)
                                chunk_frame_scores = max_sim_per_query.mean(dim=0)  # (chunk_size,)
                                
                                frame_scores.append(chunk_frame_scores)
                                
                                # Clean up intermediate tensors
                                del similarity, max_sim_per_query, chunk_frame_scores

                            logits_list.append(torch.cat(frame_scores, dim=0))
                        
                        logits = torch.stack(logits_list)  # (batch_size, block_num)
                    else:
                        # Average-vector similarity for local window frames
                        query_avg = query_tokens.mean(dim=1)  # (batch_size, D)
                        frame_avg = global_k.mean(dim=2)  # (batch_size, block_num, D)
                        logits = torch.einsum('bd,bfd->bf', query_avg, frame_avg)  # (batch_size, block_num)
            else:  # The local window is already filled, but the number of input frames is less than 'topk'.
                ret = [list(range(len(self.global_blocks[0]))) for _ in range(self.num_units)]
        else:
            # Use hybrid approach: fast average-based pre-filtering + maxSim for refinement
            logits = self._calc_hybrid_similarity(query_tokens) if self.use_hybrid_similarity else self._calc_avg_similarity(query_tokens)

        if logits is not None:
            self.similarity = logits
            assert self.topk % self.chunk_size == 0
            remainder_size = logits.shape[1] % self.chunk_size
            chunked_logits = logits[:, :logits.shape[1]-remainder_size].reshape(self.num_units, -1, self.chunk_size).mean(dim=-1)  # (batch_size, block_num // chunk_size)
            if remainder_size > 0:
                remainder_logits = logits[:, -remainder_size:].mean(dim=-1, keepdim=True)  # (batch_size, 1)
                chunked_logits = torch.cat([chunked_logits, remainder_logits], dim=1)
            ret = chunked_logits.topk(self.topk//self.chunk_size, dim=1).indices
            ret = ret.sort(dim=1)[0][:, :, None]  # (batch_size, topk//chunk_size, 1)
            ret = ret * self.chunk_size + torch.arange(self.chunk_size, device=ret.device)[None, None, :]  # (batch_size, topk//chunk_size, chunk_size)
            ret = ret.reshape(self.num_units, -1)  # (batch_size, topk)
            ret = ret.cpu().tolist()

            # NOTE: The last chunk might cause an index overflow
            for u in range(self.num_units):
                ret[u] = list(filter(lambda idx: idx < logits.shape[1], ret[u]))

        return ret

    def _calc_hybrid_similarity(self, query_tokens):
        """
        Hybrid similarity calculation: fast average-based pre-filtering + maxSim for top candidates
        
        Args:
            query_tokens: (batch_size, length, num_heads * dim_head)
            
        Returns:
            logits: (batch_size, block_num) similarity scores
        """
        # Step 1: Fast pre-filtering using average vectors (original approach)
        query_avg = query_tokens.mean(dim=1)  # (batch_size, num_heads * dim_head)
        
        # Calculate cosine similarity with stored average vectors
        avg_logits = torch.stack([self.block_k[u].get_cosine_similarity(query_avg[u]) for u in range(self.num_units)])  # (batch_size, block_num)
        
        # Step 2: Select top candidates for maxSim refinement (e.g., top 3x of what we need)
        pre_filter_k = min(self.topk * 3, avg_logits.size(1))
        _, top_indices = avg_logits.topk(pre_filter_k, dim=1)  # (batch_size, pre_filter_k)
        
        # Step 3: Apply maxSim only to top candidates to refine the ranking
        refined_logits = torch.zeros_like(avg_logits)
        
        for u in range(self.num_units):
            for idx in top_indices[u]:
                # Get detailed tokens for this frame from CPU storage
                detailed_k = self.global_blocks[u][idx].detailed_k.to(query_tokens.device)  # (block_size, hidden_size)
                
                # Calculate maxSim similarity for this specific frame
                query_u = query_tokens[u]  # (length, hidden_size)
                
                # Memory-efficient maxSim calculation
                similarity = torch.matmul(query_u, detailed_k.T)  # (length, block_size)
                max_sim_per_query = similarity.max(dim=-1)[0]  # (length,)
                frame_score = max_sim_per_query.max(dim=0)[0]  # scalar
                
                refined_logits[u, idx] = frame_score
                
                # Clean up
                del detailed_k, similarity, max_sim_per_query
            
            # For non-top candidates, use the average similarity
            mask = torch.ones(avg_logits.size(1), dtype=torch.bool, device=avg_logits.device)
            mask[top_indices[u]] = False
            refined_logits[u, mask] = avg_logits[u, mask]
        
        return refined_logits

    def _calc_avg_similarity(self, query_tokens):
        """
        Original average-vector similarity calculation across all blocks.
        Args:
            query_tokens: (batch_size, length, num_heads * dim_head)
        Returns:
            logits: (batch_size, block_num)
        """
        query_avg = query_tokens.mean(dim=1)
        avg_logits = torch.stack([
            self.block_k[u].get_cosine_similarity(query_avg[u]) for u in range(self.num_units)
        ])
        return avg_logits

    # load init KV
    def get_global_hidden_and_mask(self, exc_length):
        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        global_remainder_ed = self._global_remainder_ed + exc_length
        global_remainder_st = self._global_remainder_st
        global_remainder_len = global_remainder_ed - global_remainder_st

        # prepare init KV-Cache until it's full
        if not self.init_exc and global_remainder_len > self.n_local:
            global_k = self.global_remainder[0]
            global_v = self.global_remainder[1]

            append_init_len = min(
                self.n_init - self.init_k.size(-2),
                global_remainder_len - self.n_local
            )
            self.init_k = torch.cat(
                (self.init_k, global_k[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            self.init_v = torch.cat(
                (self.init_v, global_v[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
            )
            global_remainder_st += append_init_len
            global_remainder_len -= append_init_len

            if self.init_k.size(-2) == self.n_init:
                self.init_exc = True  # init KV-Cache is full

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st

        # load init KV
        init_st = 0
        init_ed = init_st + self.init_k.size(-2)
        if self.global_buffer_init_st != init_st or self.global_buffer_init_ed != init_ed:  # init KV haven't been loaded into global_h_kv
            global_h_k[:, :, init_st: init_ed, :].copy_(self.init_k, non_blocking=True)
            global_h_v[:, :, init_st: init_ed, :].copy_(self.init_v, non_blocking=True)

        self.global_buffer_init_st = init_st
        self.global_buffer_init_ed = init_ed

        global_h_k = global_h_k[:, :, :init_ed, :]
        global_h_v = global_h_v[:, :, :init_ed, :]

        return global_h_k, global_h_v

    def _append(
        self,
        local_q, local_k, local_v, global_q,
    ):
        """calculate attention results 

        Args:
            local_q (_type_): (batch_size, num_heads, length, dim_head)
            local_k (_type_): (batch_size, num_heads, length, dim_head)
            local_v (_type_): (batch_size, num_heads, length, dim_head)
            global_q (_type_): (batch_size, num_heads, length, dim_head)

        Returns:
            chunk_o: (batch_size, num_heads, length, dim_head)
        """

        # apply RoPE to input QKV
        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v

        # input Q attends to input + local KV
        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(
            local_h_q, local_h_k, local_h_v, 
            get_score=False, sliding_window=self.n_local
        )

        # load init KV
        with torch.cuda.stream(GLOBAL_STREAM):
            global_h_q = global_q
            global_h_k, global_h_v = self.get_global_hidden_and_mask(exc_length=global_q.size(-2))

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        # input Q attends to init KV
        attn.append(
            global_h_q, global_h_k, global_h_v, 
            end=True,  # the final append operation
            get_score=False, 
            sliding_window=None,
            complement_sliding_window=True,
        )

        o, _ = attn.get_result()

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        return o.view((self.batch_size, self.num_heads, -1, self.dim_head))

    def _append_global(
        self
    ):
        """offload context memory
        """

        global_remainder_ed = self._global_remainder_ed
        global_remainder_st = self._global_remainder_st

        global_remainder_len = global_remainder_ed - global_remainder_st

        # offload context KV to CPU
        if self.init_exc:
            assert global_remainder_len % self.block_size == 0, f'global_remainder_len: {global_remainder_len}, block_size: {self.block_size}'
            while global_remainder_len > 0:
                global_remainder_len -= self.block_size
                # Compute representative keys for decision
                global_block_k = self.global_remainder[0][:, :, global_remainder_st:global_remainder_st + self.block_size, :]
                global_block_k = self._from_group_kv(global_block_k)  # (batch_size, num_heads, length, dim_head)
                global_block_k_avg = global_block_k.mean(dim=-2, keepdim=False)  # (batch_size, num_heads, dim_head)
                global_block_k_avg_flat = global_block_k_avg.reshape(self.num_units, -1)  # (batch_size, num_heads * dim_head)

                # Context KV-Cache
                for u in range(self.num_units):
                    self.global_blocks[u].append((
                        MemoryUnit(
                            (
                                self.global_remainder[0][u, :, global_remainder_st:global_remainder_st + self.block_size, :],
                                self.global_remainder[1][u, :, global_remainder_st:global_remainder_st + self.block_size, :]
                            ),
                            self.cuda_cache,
                            False,
                            self.pin_memory
                        )
                    ))

                # Store both average vector for quick similarity and detailed tokens for maxSim when needed
                global_block_k = self.global_remainder[0][:, :, global_remainder_st:global_remainder_st + self.block_size, :]
                global_block_k = self._from_group_kv(global_block_k)  # (batch_size, num_heads, length, dim_head)

                # Store average vector for memory efficiency (fallback to original approach)
                global_block_k_avg = global_block_k.mean(dim=-2, keepdim=False)  # (batch_size, num_heads, dim_head)
                global_block_k_avg = global_block_k_avg.reshape(self.num_units, -1)  # (batch_size, num_heads * dim_head)
                global_block_k_avg = global_block_k_avg[:, None, :]  # (batch_size, 1, num_heads * dim_head)
                for u in range(self.num_units):
                    self.block_k[u].append(global_block_k_avg[u])

                # Store detailed tokens in MemoryUnit for maxSim calculation when needed
                global_block_k_detailed = global_block_k.transpose(1, 2)  # (batch_size, block_size, num_heads, dim_head)
                global_block_k_detailed = global_block_k_detailed.reshape(self.num_units, self.block_size, -1)  # (batch_size, block_size, num_heads * dim_head)
                
                # Store detailed tokens in the MemoryUnit for on-demand maxSim
                for u in range(self.num_units):
                    # Add detailed frame tokens to the memory unit for maxSim retrieval
                    if not hasattr(self.global_blocks[u][-1], 'detailed_k'):
                        self.global_blocks[u][-1].detailed_k = global_block_k_detailed[u].cpu()  # Store on CPU to save GPU memory
                
                self.num_global_block += 1
                global_remainder_st += self.block_size

        self._global_remainder_ed = global_remainder_ed
        self._global_remainder_st = global_remainder_st

    def append(
        self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
    ):
        # Pre-allocate GPU Memory.
        if not self.initialized:
            self.init(
                local_q, local_k, local_v,
                global_q, global_k, global_v
            )

        input_length = local_q.size(-2)
        
        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        # append local KV
        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        kv_length = self.local_k.size(-2)

        # append global remainder
        with torch.cuda.stream(GLOBAL_STREAM):
            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)

            self.global_remainder = (
                torch.cat((self.global_remainder[0], global_k), dim=-2),
                torch.cat((self.global_remainder[1], global_v), dim=-2),
            )

        # apply RoPE to global_q
        with torch.cuda.stream(GLOBAL_STREAM):
            # Save query_states before RoPE if flag is set and not in retrieval mode
            if self.save_query_states_flag and not self.to_retrieve:
                self.saved_query_states = global_q.clone().detach()
            
            global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(
                global_q, self.n_local
            )

        o_list = []
        for st in range(0, input_length, self.exc_block_size):  # Process the input tokens in blocks.
            ed = min(st + self.exc_block_size, input_length)

            # calculate attention results
            kv_st = max(kv_length + st - input_length - self.n_local, 0)
            kv_ed = kv_length + ed - input_length
            chunk_o = self._append(
                local_q[:, :, st:ed, :],
                self.local_k[:, :, kv_st: kv_ed, :],
                self.local_v[:, :, kv_st: kv_ed, :],
                global_q[:, :, st:ed, :],
            )
            o_list.append(chunk_o)

            # offload context memory
            with torch.cuda.stream(GLOBAL_STREAM):
                self._append_global()

            if self.async_global_stream:
                torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        self.length += input_length

        # restrict the length of local KV-cache to self.n_local
        if self.local_k.size(-2) >= self.n_local:
            self.local_k = self.local_k[:, :, -self.n_local:, :]
            self.local_v = self.local_v[:, :, -self.n_local:, :]

        # update global remainder
        assert self._global_remainder_ed == self.global_remainder[0].size(-2)
        assert not self.init_exc or self._global_remainder_st == self._global_remainder_ed, f'self.init_exc: {self.init_exc}, global_remainder_st: {self._global_remainder_st}, global_remainder_ed: {self._global_remainder_ed}'
        with torch.cuda.stream(GLOBAL_STREAM):
            self.global_remainder = (
                self.global_remainder[0][:, :, self._global_remainder_st:, :],
                self.global_remainder[1][:, :, self._global_remainder_st:, :]
            )

        ret = torch.cat(o_list, dim=-2)
        
        return ret
    
    def size(self, *args, **kwargs):
        return self.length

    def calculate_cpu_memory(self):
        memory = 0
        for u in range(self.num_units):
            for block in self.global_blocks[u]:
                memory += block.calculate_cpu_memory()
        return memory

class DecoupledKVManager:
    """
    Pure KV storage and retrieval manager (decoupled from attention/ROPE).
    Maintains:
      - init_k/v: first n_init tokens
      - local_k/v: last n_local tokens (sliding window)
      - offloaded context blocks on CPU with representative vectors and detailed tokens
    Retrieval uses pre-RoPE query tokens passed from attention via cache_kwargs.
    """
    def __init__(
        self,
        n_init: int,
        n_local: int,
        block_size: int,
        max_cached_block: int,
        topk: int,
        chunk_size: int,
        async_global_stream: bool = True,
        pin_memory: bool = True,
        use_hybrid_similarity: bool = True,
    ):
        self.n_init = int(n_init)
        self.n_local = int(n_local) if n_local is not None else 0
        self.block_size = int(block_size)
        self.max_cached_block = int(max_cached_block)
        self.topk = int(topk)
        self.chunk_size = int(chunk_size)
        self.async_global_stream = bool(async_global_stream)
        self.pin_memory = bool(pin_memory)
        self.use_hybrid_similarity = bool(use_hybrid_similarity)

        # Parameter sanity checks (align with ContextManager expectations)
        assert self.block_size > 0, "block_size must be a positive integer"
        assert self.n_init >= 0, "n_init must be non-negative"
        assert self.n_local >= 0, "n_local must be non-negative"
        assert self.chunk_size >= 1, "chunk_size must be >= 1"
        assert self.max_cached_block >= 0, "max_cached_block must be non-negative"
        if self.topk > 0:
            assert self.topk % self.chunk_size == 0, "topk must be divisible by chunk_size"
        if self.n_local > 0:
            assert self.block_size <= self.n_local, "block_size must be <= n_local when a local window is used"

        self.initialized = False
        self.init_exc = False
        self.num_units = None
        self.unit_size_kv = None
        self.dim_head = None
        self.dtype = None
        self.device = None

        self.init_k = None
        self.init_v = None
        self.local_k = None
        self.local_v = None
        self._total_ingested = 0

        self.global_blocks = []  # List[List[MemoryUnit]]
        self.cached_blocks = []  # List[Dict[int, int]]
        self.num_global_block = 0
        self.block_k = []  # List[VectorTensor]
        self.cuda_cache = None
        self._remainder_k = None
        self._remainder_v = None

        # Active base mode: after a retrieval, persist base KV = init + retrieved frames
        # and append subsequent KV to this base (no sliding-window truncation) until deactivated.
        self.active_mode = False
        self.active_base_k = None
        self.active_base_v = None
        self.appended_k = None
        self.appended_v = None

        self.add_cache = True
        self.to_retrieve = False
        self.retrieved_block_indices = None

        # Query states saving for entropy-adaptive retrieval
        self.saved_query_states: Optional[torch.Tensor] = None
        self.save_query_states_flag: bool = False
        self.use_saved_query_states_flag: bool = False

        global GLOBAL_STREAM
        if self.async_global_stream and GLOBAL_STREAM is None:
            GLOBAL_STREAM = torch.cuda.Stream()

    def _lazy_init(self, k: torch.Tensor, v: torch.Tensor):
        assert k.is_cuda and v.is_cuda
        assert k.dim() == 4 and v.dim() == 4, "k and v must be 4D tensors (B, n_kv_heads, L, Dh)"
        assert (
            k.size(0) == v.size(0)
            and k.size(1) == v.size(1)
            and k.size(3) == v.size(3)
        ), "k and v must share batch size, n_kv_heads and head_dim"
        self.num_units = k.size(0)
        self.unit_size_kv = k.size(1)
        self.dim_head = k.size(3)
        self.dtype = k.dtype
        self.device = k.device
        self.init_k = torch.empty((self.num_units, self.unit_size_kv, 0, self.dim_head), dtype=self.dtype, device=self.device)
        self.init_v = torch.empty((self.num_units, self.unit_size_kv, 0, self.dim_head), dtype=self.dtype, device=self.device)
        self.local_k = torch.empty((self.num_units, self.unit_size_kv, 0, self.dim_head), dtype=self.dtype, device=self.device)
        self.local_v = torch.empty((self.num_units, self.unit_size_kv, 0, self.dim_head), dtype=self.dtype, device=self.device)
        self.global_blocks = [[] for _ in range(self.num_units)]
        self.cached_blocks = [{} for _ in range(self.num_units)]
        self.block_k = [VectorTensor(self.unit_size_kv * self.dim_head, self.dtype, self.device) for _ in range(self.num_units)]
        self.cuda_cache = CudaCache(
            max(1, self.max_cached_block * self.num_units),
            self.unit_size_kv * self.block_size * self.dim_head * 2,
            self.dtype,
        )
        self._remainder_k = torch.empty((self.num_units, self.unit_size_kv, 0, self.dim_head), dtype=self.dtype, device=self.device)
        self._remainder_v = torch.empty((self.num_units, self.unit_size_kv, 0, self.dim_head), dtype=self.dtype, device=self.device)
        self.initialized = True

    def set_retrieval(self):
        self.to_retrieve = True

    def reset_retrieval(self):
        self.to_retrieve = False
        self.retrieved_block_indices = None

    def set_retrieved_block_indices(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().tolist()
        self.retrieved_block_indices = indices

    def save_query_states(self, save: bool):
        """Set flag to save query states during forward pass."""
        self.save_query_states_flag = bool(save)

    def use_saved_query_states(self, use: bool):
        """Set flag to use saved query states for retrieval instead of new ones."""
        self.use_saved_query_states_flag = bool(use)

    def clear_saved_query_states(self):
        """Clear saved query states and reset flags."""
        self.saved_query_states = None
        self.save_query_states_flag = False
        self.use_saved_query_states_flag = False

    # Activate persistent base = (init + retrieved frames)
    def activate_base(self, k_out: torch.Tensor, v_out: torch.Tensor):
        # Expect shape (B, n_kv_heads, L, Dh)
        self.active_base_k = k_out.contiguous()
        self.active_base_v = v_out.contiguous()
        # Reset appended tails
        self.appended_k = torch.empty(
            (self.num_units, self.unit_size_kv, 0, self.dim_head),
            dtype=self.dtype,
            device=self.device,
        )
        self.appended_v = torch.empty(
            (self.num_units, self.unit_size_kv, 0, self.dim_head),
            dtype=self.dtype,
            device=self.device,
        )
        self.active_mode = True

    def deactivate_base(self):
        self.active_mode = False
        self.active_base_k = None
        self.active_base_v = None
        self.appended_k = None
        self.appended_v = None

    def _offload_blocks_if_ready(self):
        # Only offload after init is filled
        if not self.init_exc:
            return
        rem_len = self._remainder_k.size(-2)
        if rem_len < self.block_size:
            return
        num_full = rem_len // self.block_size
        take_len = num_full * self.block_size
        k_full = self._remainder_k[:, :, :take_len, :]
        v_full = self._remainder_v[:, :, :take_len, :]
        # Ensure we only offload full frames aligned to block_size
        assert k_full.size(-2) % self.block_size == 0, f'k_full_len: {k_full.size(-2)}, block_size: {self.block_size}'
        if self.async_global_stream:
            # Ensure GLOBAL_STREAM starts after current stream enqueues preceding ops
            cur_stream = torch.cuda.current_stream()
            start_evt = torch.cuda.Event()
            start_evt.record(cur_stream)
            GLOBAL_STREAM.wait_event(start_evt)
            with torch.cuda.stream(GLOBAL_STREAM):
                for blk_idx in range(num_full):
                    st = blk_idx * self.block_size
                    ed = st + self.block_size
                    for u in range(self.num_units):
                        mu = MemoryUnit((k_full[u, :, st:ed, :], v_full[u, :, st:ed, :]), self.cuda_cache, False, self.pin_memory)
                        self.global_blocks[u].append(mu)
                    k_blk = k_full[:, :, st:ed, :].permute(0, 2, 1, 3).reshape(self.num_units, self.block_size, -1)
                    k_avg = k_blk.mean(dim=1)  # (B, D)
                    for u in range(self.num_units):
                        self.block_k[u].append(k_avg[u][None, :].contiguous())
                    k_detailed = k_blk  # (B, block_size, D)
                    for u in range(self.num_units):
                        if not hasattr(self.global_blocks[u][-1], 'detailed_k'):
                            self.global_blocks[u][-1].detailed_k = k_detailed[u].contiguous().to('cpu', non_blocking=True)
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)
        else:
            for blk_idx in range(num_full):
                st = blk_idx * self.block_size
                ed = st + self.block_size
                for u in range(self.num_units):
                    mu = MemoryUnit((k_full[u, :, st:ed, :], v_full[u, :, st:ed, :]), self.cuda_cache, False, self.pin_memory)
                    self.global_blocks[u].append(mu)
                k_blk = k_full[:, :, st:ed, :].permute(0, 2, 1, 3).reshape(self.num_units, self.block_size, -1)
                k_avg = k_blk.mean(dim=1)  # (B, D)
                for u in range(self.num_units):
                    self.block_k[u].append(k_avg[u][None, :].contiguous())
                k_detailed = k_blk  # (B, block_size, D)
                for u in range(self.num_units):
                    if not hasattr(self.global_blocks[u][-1], 'detailed_k'):
                        self.global_blocks[u][-1].detailed_k = k_detailed[u].contiguous().to('cpu', non_blocking=True)
        self.num_global_block += num_full
        if take_len > 0:
            self._remainder_k = self._remainder_k[:, :, take_len:, :].contiguous()
            self._remainder_v = self._remainder_v[:, :, take_len:, :].contiguous()

    def _promote_init_if_needed(self):
        if self.init_exc:
            return
        init_len = self.init_k.size(-2)
        rem_len = self._remainder_k.size(-2)
        if rem_len <= self.n_local:
            return
        promote_len = min(self.n_init - init_len, rem_len - self.n_local)
        if promote_len <= 0:
            return
        # Move from front of remainder into end of init
        self.init_k = torch.cat([self.init_k, self._remainder_k[:, :, :promote_len, :]], dim=-2)
        self.init_v = torch.cat([self.init_v, self._remainder_v[:, :, :promote_len, :]], dim=-2)
        self._remainder_k = self._remainder_k[:, :, promote_len:, :].contiguous()
        self._remainder_v = self._remainder_v[:, :, promote_len:, :].contiguous()
        if self.init_k.size(-2) >= self.n_init:
            self.init_k = self.init_k[:, :, :self.n_init, :].contiguous()
            self.init_v = self.init_v[:, :, :self.n_init, :].contiguous()
            self.init_exc = True

    def ingest_kv(self, k: torch.Tensor, v: torch.Tensor):
        if not self.initialized:
            self._lazy_init(k, v)
        # In retrieval mode, do NOT record incoming KV into any storage
        if self.to_retrieve:
            if os.environ.get('REKV_DEBUG', '0') == '1':
                try:
                    print(f"[ReKV][ingest] skip store due to retrieval mode: add_L={k.size(-2)}")
                except Exception:
                    pass
            return
        token_len = k.size(-2)
        self._total_ingested += token_len
        if os.environ.get('REKV_DEBUG', '0') == '1':
            try:
                print(f"[ReKV][ingest] active={self.active_mode} add_L={token_len} init_L={self.init_k.size(-2)} local_L={self.local_k.size(-2)} rem_L={self._remainder_k.size(-2)}")
            except Exception:
                pass
        k_tail = k
        v_tail = v
        # If in active mode, append new tokens to appended_* and skip sliding window truncation
        if self.active_mode:
            if k_tail.size(-2) > 0:
                self.appended_k = torch.cat([self.appended_k, k_tail], dim=-2)
                self.appended_v = torch.cat([self.appended_v, v_tail], dim=-2)
                if os.environ.get('REKV_DEBUG', '0') == '1':
                    try:
                        base_L = self.active_base_k.size(-2) if self.active_base_k is not None else 0
                        app_L = self.appended_k.size(-2)
                        print(f"[ReKV][ingest] appended_L={app_L} base_L={base_L} total_L={base_L+app_L}")
                    except Exception:
                        pass
            # Do NOT merge into global/local while active
            return
        if k_tail.size(-2) > 0:
            self.local_k = torch.cat([self.local_k, k_tail], dim=-2)
            self.local_v = torch.cat([self.local_v, v_tail], dim=-2)
            if self.n_local > 0 and self.local_k.size(-2) > self.n_local:
                self.local_k = self.local_k[:, :, -self.n_local:, :].contiguous()
                self.local_v = self.local_v[:, :, -self.n_local:, :].contiguous()
        if k_tail.size(-2) > 0:
            self._remainder_k = torch.cat([self._remainder_k, k_tail], dim=-2)
            self._remainder_v = torch.cat([self._remainder_v, v_tail], dim=-2)
            # Promote earliest remainder tokens into init if window exceeded
            self._promote_init_if_needed()
            # After init is filled, offload full blocks from remainder head
            self._offload_blocks_if_ready()

    def _project_query_to_kv_dim(self, query_tokens: torch.Tensor) -> torch.Tensor:
        """
        Project query tokens from shape (B, H, L, Dh) to KV head dimension (B, L, n_kv*Dh)
        by grouping heads: average over groups where group = H // n_kv.
        If not divisible, slice first n_kv heads.
        """
        assert query_tokens.dim() == 4
        B, H, L, Dh = query_tokens.shape
        n_kv = self.unit_size_kv
        q = query_tokens.transpose(1, 2).contiguous()  # (B, L, H, Dh)
        if H == n_kv:
            q_kv = q
        elif H % n_kv == 0:
            group = H // n_kv
            q_kv = q.view(B, L, n_kv, group, Dh).mean(dim=3)
        else:
            # Fallback: take first n_kv heads
            take = min(H, n_kv)
            pad = n_kv - take
            q_kv = q[:, :, :take, :]
            if pad > 0:
                pad_tensor = torch.zeros((B, L, pad, Dh), dtype=q.dtype, device=q.device)
                q_kv = torch.cat([q_kv, pad_tensor], dim=2)
        return q_kv.reshape(B, L, n_kv * Dh)

    def _calc_avg_similarity(self, query_tokens: torch.Tensor) -> torch.Tensor:
        # Average-vector similarity across all blocks
        B, H, L, Dh = query_tokens.shape
        q_proj = self._project_query_to_kv_dim(query_tokens).float()  # (B, L, n_kv*Dh)
        avg_logits = torch.stack([
            self.block_k[u].get_cosine_similarity(q_proj[u].mean(dim=0)) for u in range(B)
        ])
        return avg_logits

    def _calc_hybrid_similarity(self, query_tokens: torch.Tensor) -> torch.Tensor:
        # Fast average prefilter + maxSim refinement on shortlisted frames
        B, H, L, Dh = query_tokens.shape
        q_proj = self._project_query_to_kv_dim(query_tokens).float()  # (B, L, n_kv*Dh)
        avg_logits = torch.stack([
            self.block_k[u].get_cosine_similarity(q_proj[u].mean(dim=0)) for u in range(B)
        ])
        if self.num_global_block == 0:
            return avg_logits
        pre_filter_k = min(self.topk * 3, avg_logits.size(1))
        _, top_idx = avg_logits.topk(pre_filter_k, dim=1)
        refined = avg_logits.clone()
        for u in range(B):
            for idx in top_idx[u]:
                idx_int = int(idx.item())
                if idx_int >= len(self.global_blocks[u]):
                    continue
                detailed_k = self.global_blocks[u][idx_int].detailed_k.to(q_proj.device, dtype=q_proj.dtype, non_blocking=True)  # (block_size, n_kv*Dh)
                sim = torch.matmul(q_proj[u], detailed_k.T)  # (L, block_size)
                frame_score = sim.max(dim=-1)[0].max(dim=0)[0]
                refined[u, idx_int] = frame_score
        return refined

    def _select_topk_blocks(self, query_tokens: torch.Tensor) -> List[List[int]]:
        if self.topk > 0:
            assert self.topk % self.chunk_size == 0, "topk must be divisible by chunk_size"
        if self.num_global_block <= self.topk:
            return [list(range(self.num_global_block)) for _ in range(self.num_units)]
        logits = self._calc_hybrid_similarity(query_tokens) if self.use_hybrid_similarity else self._calc_avg_similarity(query_tokens)
        if self.chunk_size <= 1:
            ret = logits.topk(self.topk, dim=1).indices
            return [ret[u].tolist() for u in range(self.num_units)]
        remainder_size = logits.shape[1] % self.chunk_size
        chunked = logits[:, : logits.shape[1] - remainder_size].reshape(self.num_units, -1, self.chunk_size).mean(dim=-1)
        if remainder_size > 0:
            remainder_logits = logits[:, -remainder_size:].mean(dim=-1, keepdim=True)
            chunked = torch.cat([chunked, remainder_logits], dim=1)
        ret = chunked.topk(self.topk // self.chunk_size, dim=1).indices
        ret = ret.sort(dim=1)[0][:, :, None]
        ret = ret * self.chunk_size + torch.arange(self.chunk_size, device=ret.device)[None, None, :]
        ret = ret.reshape(self.num_units, -1)
        return [list(filter(lambda i: i < logits.shape[1], ret[u].tolist())) for u in range(self.num_units)]

    def get_retrieved_kv(self, query_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-init: retrieve from local remainder; Post-init: retrieve from CPU offloaded blocks
        if not self.init_exc:
            init_k_len = self.init_k.size(-2)
            rem_total_len = self._remainder_k.size(-2)
            init_len = min(self.n_init, init_k_len + rem_total_len)
            init_from_remainder = max(0, init_len - init_k_len)
            remainder_k_view = self._remainder_k[:, :, init_from_remainder:, :]
            remainder_v_view = self._remainder_v[:, :, init_from_remainder:, :]
            # Build frames from remainder (head-only full blocks)
            rem_len = remainder_k_view.size(-2)
            usable = (rem_len // self.block_size) * self.block_size
            block_num = usable // self.block_size
            # Ensure usable tokens form complete frames
            if usable > 0:
                assert usable % self.block_size == 0, f'usable: {usable}, block_size: {self.block_size}'
            if block_num <= 0:
                # No frames yet; return init only
                if init_len <= 0:
                    return self.init_k, self.init_v
                if init_from_remainder <= 0:
                    return self.init_k[:, :, :init_len, :], self.init_v[:, :, :init_len, :]
                k_init = torch.empty(
                    (self.num_units, self.unit_size_kv, init_len, self.dim_head),
                    dtype=self.dtype,
                    device=self.device,
                )
                v_init = torch.empty(
                    (self.num_units, self.unit_size_kv, init_len, self.dim_head),
                    dtype=self.dtype,
                    device=self.device,
                )
                copied = 0
                if init_k_len > 0:
                    take = min(init_k_len, init_len)
                    k_init[:, :, :take, :].copy_(self.init_k[:, :, :take, :], non_blocking=True)
                    v_init[:, :, :take, :].copy_(self.init_v[:, :, :take, :], non_blocking=True)
                    copied = take
                if init_from_remainder > 0 and copied < init_len:
                    remain_take = init_len - copied
                    k_init[:, :, copied:init_len, :].copy_(self._remainder_k[:, :, :remain_take, :], non_blocking=True)
                    v_init[:, :, copied:init_len, :].copy_(self._remainder_v[:, :, :remain_take, :], non_blocking=True)
                return k_init, v_init
            if query_tokens is None:
                # Fallback: take earliest frames up to topk
                indices = [list(range(min(self.topk, block_num))) for _ in range(self.num_units)]
            else:
                # Compute hybrid similarity over remainder frames
                B = self.num_units
                D_kv = self.unit_size_kv * self.dim_head
                q_proj = self._project_query_to_kv_dim(query_tokens).float()  # (B, L, D_kv)
                # frames: (B, block_num, block_size, D_kv)
                frames = remainder_k_view[:, :, :usable, :].permute(0, 2, 1, 3).contiguous().view(B, usable, D_kv)
                frames = frames.view(B, block_num, self.block_size, D_kv).float()
                chunk_frames = 4
                logits_list = []
                for u in range(B):
                    q_u = q_proj[u]  # (L, D_kv)
                    frame_scores_u = []
                    for start in range(0, block_num, chunk_frames):
                        end = min(start + chunk_frames, block_num)
                        chunk = frames[u, start:end]  # (c, block_size, D_kv)
                        sim = torch.einsum('ld,fbd->lfb', q_u, chunk)
                        max_sim_per_query = sim.max(dim=-1)[0]  # (L, c)
                        scores = max_sim_per_query.mean(dim=0)  # (c,)
                        frame_scores_u.append(scores)
                        del sim, max_sim_per_query, scores
                    logits_list.append(torch.cat(frame_scores_u, dim=0))  # (block_num,)
                logits = torch.stack(logits_list)  # (B, block_num)
                # chunked selection by chunk_size
                if self.chunk_size > 1:
                    remainder_size = logits.shape[1] % self.chunk_size
                    chunked_logits = logits[:, :logits.shape[1]-remainder_size].reshape(B, -1, self.chunk_size).mean(dim=-1)
                    if remainder_size > 0:
                        remainder_logits = logits[:, -remainder_size:].mean(dim=-1, keepdim=True)
                        chunked_logits = torch.cat([chunked_logits, remainder_logits], dim=1)
                    ret = chunked_logits.topk(self.topk // self.chunk_size, dim=1).indices
                    ret = ret.sort(dim=1)[0][:, :, None]
                    ret = ret * self.chunk_size + torch.arange(self.chunk_size, device=ret.device)[None, None, :]
                    indices = [list(filter(lambda i: i < logits.shape[1], ret[u].reshape(-1).tolist())) for u in range(B)]
                else:
                    ret = logits.topk(min(self.topk, logits.size(1)), dim=1).indices
                    indices = [ret[u].tolist() for u in range(B)]
            # Ensure ascending order for consistent temporal layout
            indices = [sorted(idx) for idx in indices]
            if os.environ.get('REKV_DEBUG', '0') == '1':
                try:
                    print(f"[ReKV][retrieve-preinit] block_num={block_num} init_L={init_len} sel_cnt={[len(x) for x in indices]} sample0={indices[0][:min(10, len(indices[0]))] if indices else []}")
                except Exception:
                    pass
            # Build outputs: init + selected remainder frames (async if enabled)
            select_counts = [len(idx) for idx in indices]
            max_sel = max(select_counts) if select_counts else 0
            total_len = init_len + max_sel * self.block_size
            if self.async_global_stream:
                with torch.cuda.stream(GLOBAL_STREAM):
                    k_out = torch.empty((self.num_units, self.unit_size_kv, total_len, self.dim_head), dtype=self.dtype, device=self.device)
                    v_out = torch.empty((self.num_units, self.unit_size_kv, total_len, self.dim_head), dtype=self.dtype, device=self.device)
                    if init_len > 0:
                        copied = 0
                        if init_k_len > 0:
                            take = min(init_k_len, init_len)
                            k_out[:, :, :take, :].copy_(self.init_k[:, :, :take, :], non_blocking=True)
                            v_out[:, :, :take, :].copy_(self.init_v[:, :, :take, :], non_blocking=True)
                            copied = take
                        if init_from_remainder > 0 and copied < init_len:
                            remain_take = init_len - copied
                            k_out[:, :, copied:init_len, :].copy_(self._remainder_k[:, :, :remain_take, :], non_blocking=True)
                            v_out[:, :, copied:init_len, :].copy_(self._remainder_v[:, :, :remain_take, :], non_blocking=True)
                    for u in range(self.num_units):
                        st = init_len
                        for b_idx in indices[u]:
                            rem_st = init_from_remainder + b_idx * self.block_size
                            rem_ed = rem_st + self.block_size
                            if rem_ed > self._remainder_k.size(-2):
                                break
                            k_out[u, :, st:st+self.block_size, :].copy_(self._remainder_k[u, :, rem_st:rem_ed, :], non_blocking=True)
                            v_out[u, :, st:st+self.block_size, :].copy_(self._remainder_v[u, :, rem_st:rem_ed, :], non_blocking=True)
                            st += self.block_size
                torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)
            else:
                k_out = torch.empty((self.num_units, self.unit_size_kv, total_len, self.dim_head), dtype=self.dtype, device=self.device)
                v_out = torch.empty((self.num_units, self.unit_size_kv, total_len, self.dim_head), dtype=self.dtype, device=self.device)
                if init_len > 0:
                    copied = 0
                    if init_k_len > 0:
                        take = min(init_k_len, init_len)
                        k_out[:, :, :take, :].copy_(self.init_k[:, :, :take, :], non_blocking=True)
                        v_out[:, :, :take, :].copy_(self.init_v[:, :, :take, :], non_blocking=True)
                        copied = take
                    if init_from_remainder > 0 and copied < init_len:
                        remain_take = init_len - copied
                        k_out[:, :, copied:init_len, :].copy_(self._remainder_k[:, :, :remain_take, :], non_blocking=True)
                        v_out[:, :, copied:init_len, :].copy_(self._remainder_v[:, :, :remain_take, :], non_blocking=True)
                for u in range(self.num_units):
                    st = init_len
                    for b_idx in indices[u]:
                        rem_st = init_from_remainder + b_idx * self.block_size
                        rem_ed = rem_st + self.block_size
                        if rem_ed > self._remainder_k.size(-2):
                            break
                        k_out[u, :, st:st+self.block_size, :].copy_(self._remainder_k[u, :, rem_st:rem_ed, :], non_blocking=True)
                        v_out[u, :, st:st+self.block_size, :].copy_(self._remainder_v[u, :, rem_st:rem_ed, :], non_blocking=True)
                        st += self.block_size
            self.set_retrieved_block_indices(indices)
            # Ensure retrieved KV length does not exceed n_init + n_local
            assert k_out.size(-2) <= self.n_init + self.n_local, f'retrieved length {k_out.size(-2)} exceeds n_init + n_local {self.n_init + self.n_local}'
            return k_out, v_out

        # Post-init: retrieve from CPU offloaded blocks
        init_len = self.init_k.size(-2)
        if query_tokens is not None:
            indices = self._select_topk_blocks(query_tokens)
        else:
            if self.retrieved_block_indices is None:
                indices = [list(range(min(self.topk, self.num_global_block))) for _ in range(self.num_units)]
            else:
                indices = self.retrieved_block_indices
        # Ensure ascending order for consistent temporal layout
        indices = [sorted(idx) for idx in indices]
        # Validate retrieved indices are within available global blocks
        for u in range(self.num_units):
            if len(indices[u]) > 0:
                assert indices[u][-1] < self.num_global_block, f'{indices[u][-1]}, {self.num_global_block}'
        if os.environ.get('REKV_DEBUG', '0') == '1':
            try:
                print(f"[ReKV][retrieve-postinit] global_blocks={self.num_global_block} init_L={init_len} sel_cnt={[len(x) for x in indices]} sample0={indices[0][:min(10, len(indices[0]))] if indices else []}")
            except Exception:
                pass
        select_counts = [len(idx) for idx in indices]
        max_sel = max(select_counts) if select_counts else 0
        total_len = init_len + max_sel * self.block_size
        if self.async_global_stream:
            with torch.cuda.stream(GLOBAL_STREAM):
                k_out = torch.empty((self.num_units, self.unit_size_kv, total_len, self.dim_head), dtype=self.dtype, device=self.device)
                v_out = torch.empty((self.num_units, self.unit_size_kv, total_len, self.dim_head), dtype=self.dtype, device=self.device)
                if init_len > 0:
                    k_out[:, :, :init_len, :].copy_(self.init_k, non_blocking=True)
                    v_out[:, :, :init_len, :].copy_(self.init_v, non_blocking=True)
                for u in range(self.num_units):
                    st = init_len
                    for b_idx in indices[u]:
                        if b_idx >= len(self.global_blocks[u]):
                            continue
                        target_k = k_out[u, :, st : st + self.block_size, :]
                        target_v = v_out[u, :, st : st + self.block_size, :]
                        # Avoid allocating a GPU cache slot: copy directly from CPU-stored block
                        mu = self.global_blocks[u][b_idx]
                        src_k, src_v = mu.cpu_data  # shapes: (n_kv_heads, block_size, Dh)
                        target_k.copy_(src_k, non_blocking=True)
                        target_v.copy_(src_v, non_blocking=True)
                        st += self.block_size
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)
        else:
            k_out = torch.empty((self.num_units, self.unit_size_kv, total_len, self.dim_head), dtype=self.dtype, device=self.device)
            v_out = torch.empty((self.num_units, self.unit_size_kv, total_len, self.dim_head), dtype=self.dtype, device=self.device)
            if init_len > 0:
                k_out[:, :, :init_len, :].copy_(self.init_k, non_blocking=True)
                v_out[:, :, :init_len, :].copy_(self.init_v, non_blocking=True)
            for u in range(self.num_units):
                st = init_len
                for b_idx in indices[u]:
                    if b_idx >= len(self.global_blocks[u]):
                        continue
                    target_k = k_out[u, :, st : st + self.block_size, :]
                    target_v = v_out[u, :, st : st + self.block_size, :]
                    # Avoid allocating a GPU cache slot: copy directly from CPU-stored block
                    mu = self.global_blocks[u][b_idx]
                    src_k, src_v = mu.cpu_data  # shapes: (n_kv_heads, block_size, Dh)
                    target_k.copy_(src_k, non_blocking=True)
                    target_v.copy_(src_v, non_blocking=True)
                    st += self.block_size
        # except Exception as e:
        #     breakpoint()
        self.set_retrieved_block_indices(indices)
        # Ensure retrieved KV length does not exceed n_init + n_local
        assert k_out.size(-2) <= self.n_init + self.n_local, f'retrieved length {k_out.size(-2)} exceeds n_init + n_local {self.n_init + self.n_local}'
        if os.environ.get('REKV_DEBUG', '0') == '1':
            try:
                print(f"[ReKV][retrieve] k_out={tuple(k_out.shape)} v_out={tuple(v_out.shape)}")
            except Exception:
                pass
        return k_out, v_out

    def set_block_size(self, new_block_size: int):
        """
        Dynamically adjust block_size for DecoupledKVManager. Preserves init/local/remainder
        tokens and active base, but resets offloaded blocks and representative vectors.
        """
        if not isinstance(new_block_size, int) or new_block_size <= 0:
            return
        if new_block_size == self.block_size:
            return
        if self.n_local > 0:
            assert new_block_size <= self.n_local, "new_block_size must be <= n_local when a local window is used"
        self.block_size = int(new_block_size)
        self.n_local = 144 * self.block_size
        if not self.initialized:
            # Will allocate with new block_size on first ingest
            return
        # Reset offloaded state (depends on block partitioning)
        self.global_blocks = [[] for _ in range(self.num_units)]
        self.cached_blocks = [{} for _ in range(self.num_units)]
        self.num_global_block = 0
        # Recreate representative vector stores
        self.block_k = [VectorTensor(self.unit_size_kv * self.dim_head, self.dtype, self.device) for _ in range(self.num_units)]
        # Recreate CUDA cache with new unit size
        self.cuda_cache = CudaCache(
            max(1, self.max_cached_block * self.num_units),
            self.unit_size_kv * self.block_size * self.dim_head * 2,
            self.dtype,
        )

    def get_sliding_window_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # In active mode, always return base + appended without truncation
        if self.active_mode:
            if self.appended_k is None or self.appended_k.size(-2) == 0:
                return self.active_base_k, self.active_base_v
            k_cat = torch.cat([self.active_base_k, self.appended_k], dim=-2)
            v_cat = torch.cat([self.active_base_v, self.appended_v], dim=-2)
            return k_cat, v_cat
        if self.n_local > 0:
            k_cat = torch.cat([self.init_k, self.local_k], dim=-2) if self.init_k.size(-2) > 0 else self.local_k
            v_cat = torch.cat([self.init_v, self.local_v], dim=-2) if self.init_v.size(-2) > 0 else self.local_v
            return k_cat, v_cat
        return self.local_k, self.local_v
    
    def calculate_cpu_memory(self):
        memory = 0
        for u in range(self.num_units):
            for block in self.global_blocks[u]:
                memory += block.calculate_cpu_memory()
        return memory

class ReKVDynamicCache(Cache):
    """
    HF-compatible dynamic cache wrapping per-layer ContextManager.
    Provides minimal interface used by Qwen2 attention forward:
      - get_seq_length
      - get_usable_length
      - update
      - iteration of per-layer managers and control of retrieval flags
    """
    def __init__(self, rekv_config: Optional[Dict[str, Any]], num_layers: int):
        super().__init__()
        # Store config for constructing layer managers lazily when first update comes
        self.rekv_config = rekv_config or {}
        self.num_layers = int(num_layers)
        self.layer_caches: List[Optional[DecoupledKVManager]] = [None] * self.num_layers
        self._seq_len = 0
        # mRoPE related configuration (Qwen2-VL)
        self._mrope_cfg: Dict[str, Any] = {
            "image_token_id": None,
            "video_token_id": None,
            "vision_start_token_id": None,
            "spatial_merge_size": None,
            "is_qwen2vl": False,
        }
        # Runtime attributes per forward
        self._mrope_runtime: Dict[str, Any] = {
            "image_grid_thw": None,   # Optional[torch.Tensor] (N_img, 3)
            "video_grid_thw": None,   # Optional[torch.Tensor] (N_vid, 3)
            "is_time_prompt": False,
            "num_time_tokens": 0,
        }

    def __iter__(self):
        for cm in self.layer_caches:
            if cm is None:
                # Yield a dummy object with required methods to keep compatibility with existing codepaths
                yield self
            else:
                yield cm

    # Provide sequence-like semantics for compatibility with existing code
    def __len__(self):
        return self.num_layers

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.num_layers:
            raise IndexError("ReKVDynamicCache index out of range")
        cm = self.layer_caches[idx]
        if cm is None:
            class _Dummy:
                def calculate_cpu_memory(self_inner):
                    return 0
            return _Dummy()
        return cm

    def __repr__(self):
        filled = sum(1 for x in self.layer_caches if x is not None)
        return f"ReKVDynamicCache(layers={self.num_layers}, filled={filled}, seq_len={self._seq_len})"

    def set_block_size(self, block_size: int):
        for cm in self.layer_caches:
            if cm is not None:
                cm.set_block_size(block_size)
    
    # ------------------------ mRoPE helpers ------------------------
    def set_mrope_config(
        self,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
        spatial_merge_size: int,
        is_qwen2vl: bool = True,
    ):
        self._mrope_cfg.update({
            "image_token_id": int(image_token_id),
            "video_token_id": int(video_token_id),
            "vision_start_token_id": int(vision_start_token_id),
            "vision_end_token_id": int(vision_end_token_id),
            "spatial_merge_size": int(spatial_merge_size),
            "is_qwen2vl": bool(is_qwen2vl),
        })

    def set_mrope_runtime(
        self,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        is_time_prompt: bool = False,
        num_time_tokens: int = 0,
    ):
        # Accept list/tuple as well
        def _to_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x)
        self._mrope_runtime.update({
            "image_grid_thw": _to_tensor(image_grid_thw),
            "video_grid_thw": _to_tensor(video_grid_thw),
            "is_time_prompt": bool(is_time_prompt),
            "num_time_tokens": int(num_time_tokens),
        })

    def _build_fake_input_ids_for_layer(self, cm: DecoupledKVManager, kv_seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a debug-friendly fake input_ids and attention_mask of shape (B, kv_seq_len)
        using only special token ids (vision_start, image/video) and generic text ids (0).
        This is separated from the position-id computation for easier debugging.
        """
        B = cm.num_units
        image_token_id = self._mrope_cfg.get("image_token_id", None)
        video_token_id = self._mrope_cfg.get("video_token_id", None)
        vision_start_token_id = self._mrope_cfg.get("vision_start_token_id", None)
        vision_end_token_id = self._mrope_cfg.get("vision_end_token_id", None)
        assert vision_start_token_id is not None, "vision_start_token_id is not set"
        assert vision_end_token_id is not None, "vision_end_token_id is not set"
        assert image_token_id is not None, "image_token_id is not set"
        assert video_token_id is not None, "video_token_id is not set"
        
        is_time_prompt = bool(self._mrope_runtime.get("is_time_prompt", False))
        num_time_tokens = int(self._mrope_runtime.get("num_time_tokens", 0))
        # Derive lengths
        if getattr(cm, 'active_mode', False) and cm.active_base_k is not None:
            base_len = int(cm.active_base_k.size(-2))
            app_len = int(cm.appended_k.size(-2)) if cm.appended_k is not None else 0
            init_len = cm.n_init
            retrieved_len = max(0, base_len - init_len)
        else:
            k_cur, _ = cm.get_sliding_window_kv()
            base_len = int(k_cur.size(-2))
            app_len = 0
            init_len = cm.n_init
            retrieved_len = 0
            assert kv_seq_len == base_len + app_len, f"kv_seq_len {kv_seq_len} does not match base_len {base_len} + app_len {app_len}"
        block_size = int(cm.block_size)
        # Decide how many frames are present in current KV
        if getattr(cm, 'active_mode', False) and cm.active_base_k is not None:
            # retrieval + generate: frames come from base, tail text from appended
            frames_count = int(retrieved_len // block_size) if (retrieved_len > 0 and block_size > 0) else 0
            trailing_text_len = max(app_len, kv_seq_len - base_len)
        else:
            # encode (and possibly decode text appended after frames)
            leftover_len = max(0, base_len - init_len)
            frames_count = int(leftover_len // block_size) if block_size > 0 else 0
            trailing_text_len = max(0, leftover_len - frames_count * block_size)
        # Per-frame layout
        frame_text_overhead = num_time_tokens if is_time_prompt else 0
        # Account for: time_text + vision_start + (image/video) + vision_grid
        vision_tokens_per_frame = max(0, block_size - (frame_text_overhead + 2)) if is_time_prompt else block_size
        
        # OPTIMIZED: Pre-allocate tensor instead of building Python lists
        input_ids = torch.zeros((B, kv_seq_len), dtype=torch.long, device=device)
        
        # Build sequence using vectorized operations (same for all batches)
        pos = 0
        # 1) init text (already 0, just set vision_start if needed)
        if init_len > 0:
            if not is_time_prompt and init_len > 0:
                input_ids[:, init_len - 1] = int(vision_start_token_id)
            pos = init_len
        
        # 2) retrieved frames blocks (if any)
        if frames_count > 0:
            assert video_token_id is not None, "video_token_id is not set"
            if is_time_prompt:
                # Layout per frame: [time_tokens...] [vision_start] [video_tokens...] [vision_end]
                frame_block = frame_text_overhead + 2 + vision_tokens_per_frame  # == block_size
                total_frame_tokens = frames_count * frame_block
                frame_region = (
                    input_ids[:, pos : pos + total_frame_tokens]
                    .view(B, frames_count, frame_block)
                )
                # Set vision_start / vision_end
                frame_region[:, :, frame_text_overhead] = int(vision_start_token_id)
                vision_end_idx = frame_text_overhead + 1 + vision_tokens_per_frame
                frame_region[:, :, vision_end_idx] = int(vision_end_token_id)
                # Fill video tokens (contiguous slice per frame)
                if vision_tokens_per_frame > 0:
                    frame_region[
                        :,
                        :,
                        frame_text_overhead + 1 : frame_text_overhead + 1 + vision_tokens_per_frame,
                    ] = int(video_token_id)
                pos += total_frame_tokens
            else:
                # Layout: continuous video tokens (no vision_start/end per frame)
                video_tokens_total = frames_count * block_size
                input_ids[:, pos:pos + video_tokens_total] = int(video_token_id)
                pos += video_tokens_total
        # 3) trailing text (already 0)
        pos += trailing_text_len
        
        # Sanity check
        if pos != kv_seq_len:
            breakpoint()
        assert pos == kv_seq_len, f"Position {pos} does not match kv_seq_len {kv_seq_len}"
        
        image_grid_thw = None
        video_grid_thw = None
        if frames_count > 0:
            image_grid_thw = self._mrope_runtime.get("image_grid_thw", None)
            _video_grid_thw = self._mrope_runtime.get("video_grid_thw", None)
            # breakpoint()
            if _video_grid_thw is not None:
                if is_time_prompt:
                    base_grid = _video_grid_thw[0].unsqueeze(0)
                    video_grid_thw = base_grid if frames_count == 1 else base_grid.expand(frames_count, -1)
                else:
                    # change the first dim (T) to frames_count
                    video_grid_thw = _video_grid_thw.clone()
                    video_grid_thw[0,0] = frames_count
            
        # if kv_seq_len > 10000:
        #     breakpoint()
        
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return input_ids, attention_mask, image_grid_thw, video_grid_thw

    def _compute_mrope_positions_from_input_ids(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute 3D mRoPE position ids based on a provided input_ids template and optional grids.
        This mirrors HF Qwen2-VL get_rope_index logic, adapted to cache context.
        Returns position_ids of shape (3, B, L).
        """
        spatial_merge_size = int(self._mrope_cfg.get("spatial_merge_size", 2) or 2)
        image_token_id = int(self._mrope_cfg.get("image_token_id", -1) or -1)
        video_token_id = int(self._mrope_cfg.get("video_token_id", -1) or -1)
        vision_start_token_id = int(self._mrope_cfg.get("vision_start_token_id", -1) or -1)
        B, L = input_ids.shape
        device = input_ids.device
        if (image_grid_thw is not None) or (video_grid_thw is not None):
            if image_grid_thw is not None:
                if not isinstance(image_grid_thw, torch.Tensor):
                    image_grid_thw = torch.as_tensor(image_grid_thw, device=device, dtype=torch.long)
                else:
                    image_grid_thw = image_grid_thw.to(device=device, dtype=torch.long)
            if video_grid_thw is not None:
                if not isinstance(video_grid_thw, torch.Tensor):
                    video_grid_thw = torch.as_tensor(video_grid_thw, device=device, dtype=torch.long)
                else:
                    video_grid_thw = video_grid_thw.to(device=device, dtype=torch.long)
            total_input_ids = input_ids
            position_ids = torch.ones(3, B, L, dtype=torch.long, device=device)
            image_index = 0
            video_index = 0
            for i in range(B):
                row = total_input_ids[i]
                if attention_mask is not None:
                    row = row[attention_mask[i] == 1]
                # Determine counts
                vision_start_indices = torch.argwhere(row == vision_start_token_id).squeeze(1) if vision_start_token_id >= 0 else torch.empty(0, dtype=torch.long, device=device)
                vision_tokens = row[vision_start_indices + 1] if vision_start_indices.numel() > 0 else torch.empty(0, dtype=torch.long, device=device)
                image_nums = int((vision_tokens == image_token_id).sum().item()) if image_token_id >= 0 else 0
                video_nums = int((vision_tokens == video_token_id).sum().item()) if video_token_id >= 0 else 0
                fast_video_segments = None
                fast_video_base_grid = None
                if (
                    image_nums == 0
                    and video_nums > 1
                    and video_grid_thw is not None
                    and (video_index + video_nums) <= video_grid_thw.shape[0]
                ):
                    candidate_segments = vision_start_indices[(vision_tokens == video_token_id)]
                    if candidate_segments.numel() == video_nums:
                        grids_slice = video_grid_thw[video_index : video_index + video_nums]
                        base_grid = grids_slice[0]
                        if torch.equal(grids_slice, base_grid.unsqueeze(0).expand_as(grids_slice)):
                            fast_video_segments = candidate_segments.tolist()
                            fast_video_base_grid = base_grid
                if fast_video_segments is not None:
                    grid_t = int(fast_video_base_grid[0].item())
                    grid_h = max(1, int(fast_video_base_grid[1].item()) // spatial_merge_size)
                    grid_w = max(1, int(fast_video_base_grid[2].item()) // spatial_merge_size)
                    grid_token_count = grid_t * grid_h * grid_w
                    if grid_token_count > 0:
                        grid_delta = max(grid_t, grid_h, grid_w)
                        t_idx = torch.arange(grid_t, device=device, dtype=torch.long).repeat_interleave(grid_h * grid_w)
                        h_idx = torch.arange(grid_h, device=device, dtype=torch.long).repeat_interleave(grid_w).repeat(grid_t)
                        w_idx = torch.arange(grid_w, device=device, dtype=torch.long).repeat(grid_t * grid_h)
                        base_coords = torch.stack([t_idx, h_idx, w_idx], dim=0)
                        seq_len = row.shape[0]
                        row_positions = torch.zeros(3, seq_len, dtype=torch.long, device=device)
                        cursor = 0
                        next_pos_val = 0
                        for start_idx in fast_video_segments:
                            text_end = min(seq_len, start_idx + 1)
                            if text_end > cursor:
                                text_len = text_end - cursor
                                text_positions = torch.arange(text_len, device=device, dtype=torch.long) + next_pos_val
                                row_positions[:, cursor:text_end] = text_positions
                                next_pos_val += text_len
                                cursor = text_end
                            if cursor >= seq_len:
                                break
                            write_len = min(grid_token_count, seq_len - cursor)
                            row_positions[:, cursor:cursor + write_len] = base_coords[:, :write_len] + next_pos_val
                            next_pos_val += grid_delta
                            cursor = min(seq_len, cursor + grid_token_count)
                            if cursor >= seq_len:
                                break
                        if cursor < seq_len:
                            tail_len = seq_len - cursor
                            text_positions = torch.arange(tail_len, device=device, dtype=torch.long) + next_pos_val
                            row_positions[:, cursor:] = text_positions
                        position_ids[..., i, :] = row_positions[:, :L]
                        video_index += video_nums
                        continue
                input_tokens = row.tolist()
                llm_pos_ids_list: List[torch.Tensor] = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    ed_image = input_tokens.index(image_token_id, st) if (image_token_id in input_tokens and remain_images > 0 and image_token_id >= 0) else len(input_tokens) + 1
                    ed_video = input_tokens.index(video_token_id, st) if (video_token_id in input_tokens and remain_videos > 0 and video_token_id >= 0) else len(input_tokens) + 1
                    if ed_image < ed_video:
                        if image_grid_thw is not None and image_index < image_grid_thw.shape[0]:
                            t = image_grid_thw[image_index][0]
                            h = image_grid_thw[image_index][1]
                            w = image_grid_thw[image_index][2]
                        else:
                            t = torch.tensor(1, device=device)
                            h = torch.tensor(spatial_merge_size, device=device)
                            w = torch.tensor(1 * spatial_merge_size, device=device)
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        if video_grid_thw is not None and video_index < video_grid_thw.shape[0]:
                            t = video_grid_thw[video_index][0]
                            h = video_grid_thw[video_index][1]
                            w = video_grid_thw[video_index][2]
                        else:
                            t = torch.tensor(1, device=device)
                            h = torch.tensor(spatial_merge_size, device=device)
                            w = torch.tensor(1 * spatial_merge_size, device=device)
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t = int(t.item())
                    llm_grid_h = int(h.item()) // spatial_merge_size
                    llm_grid_w = int(w.item()) // spatial_merge_size
                    text_len = ed - st
                    st_idx = int(llm_pos_ids_list[-1].max().item() + 1) if len(llm_pos_ids_list) > 0 else 0
                    if text_len > 0:
                        llm_pos_ids_list.append(torch.arange(text_len, device=device, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)
                    # Vision positions
                    t_index = torch.arange(llm_grid_t, device=device, dtype=torch.long).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h, device=device, dtype=torch.long).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w, device=device, dtype=torch.long).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    if (llm_grid_h * llm_grid_w) > 0:
                        llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                # Tail text
                if st < len(input_tokens):
                    st_idx = int(llm_pos_ids_list[-1].max().item() + 1) if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    if text_len > 0:
                        llm_pos_ids_list.append(torch.arange(text_len, device=device, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)
                if len(llm_pos_ids_list) == 0:
                    llm_positions = torch.zeros(3, len(input_tokens), device=device, dtype=torch.long)
                else:
                    llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                # assign
                position_ids[..., i, :] = llm_positions[:, :L]
            return position_ids
        # Fallback 1D ids
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(device)
            return position_ids
        else:
            position_ids = torch.arange(L, device=device, dtype=torch.long).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
            return position_ids


    def update_with_mposition_ids(self, layer_idx: int, kv_seq_len: int, q_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multimodal position ids for Qwen2-VL via cache state.
        Returns:
          - position_ids_new: (3, B, q_len) for the current step's newly added query tokens
          - kv_position_ids_full: (3, B, kv_seq_len) for the full KV used this step
        Fallbacks to simple arange when not Qwen2-VL or insufficient data/state.
        """
        cm = self.layer_caches[layer_idx]
        if cm is None or not self._mrope_cfg.get("is_qwen2vl", False):
            # Fallback simple 1D
            B = cm.num_units if cm is not None else 1
            device = cm.device if cm is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            kv_pos = torch.arange(kv_seq_len, device=device, dtype=torch.long)[None, None, :].expand(3, B, -1)
            pos_new = torch.arange(kv_seq_len - q_len, kv_seq_len, device=device, dtype=torch.long)[None, None, :].expand(3, B, -1)
            return pos_new, kv_pos
        device = cm.device
        # Step 1: Build fake input_ids (debug-separated)
        input_ids, attn_mask, image_grid_thw, video_grid_thw = self._build_fake_input_ids_for_layer(cm, kv_seq_len, device)
        # Step 2: Compute mRoPE positions using a function mirroring HF get_rope_index
        kv_pos = self._compute_mrope_positions_from_input_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attn_mask,
        )
        # if kv_seq_len > 10000:
        #     breakpoint()
        # Ensure expected shape (3, B, kv_seq_len)
        B = cm.num_units
        if kv_pos.size(2) != kv_seq_len:
            if kv_pos.size(2) < kv_seq_len:
                pad = kv_seq_len - kv_pos.size(2)
                pad_vals = kv_pos[:, :, -1:].expand(3, B, pad).contiguous()
                kv_pos = torch.cat([kv_pos, pad_vals], dim=2)
            else:
                kv_pos = kv_pos[:, :, :kv_seq_len]
        pos_new = kv_pos[:, :, -q_len:] if q_len > 0 else kv_pos.new_zeros((3, B, 0))
        return pos_new, kv_pos
    
    # Compatibility helpers for existing higher-level code
    def set_retrieval(self):
        for cm in self.layer_caches:
            if cm is not None:
                cm.set_retrieval()

    def reset_retrieval(self):
        for cm in self.layer_caches:
            if cm is not None:
                cm.reset_retrieval()

    def set_retrieved_block_indices(self, indices):
        for cm in self.layer_caches:
            if cm is not None:
                cm.set_retrieved_block_indices(indices)

    def save_query_states(self, save: bool):
        """Set flag to save query states for all layers."""
        for cm in self.layer_caches:
            if cm is not None:
                cm.save_query_states(save)

    def use_saved_query_states(self, use: bool):
        """Set flag to use saved query states for all layers."""
        for cm in self.layer_caches:
            if cm is not None:
                cm.use_saved_query_states(use)

    def clear_saved_query_states(self):
        """Clear saved query states for all layers."""
        for cm in self.layer_caches:
            if cm is not None:
                cm.clear_saved_query_states()
    
    # HF Cache API
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # During prefill, no prior cache; during decode, return current cache KV length for the layer
        cm = self.layer_caches[layer_idx]
        if cm is None:
            return 0
        # Report active base + appended when active mode is on; otherwise local window
        if getattr(cm, 'active_mode', False):
            base_len = cm.active_base_k.size(-2) if cm.active_base_k is not None else 0
            app_len = cm.appended_k.size(-2) if getattr(cm, 'appended_k', None) is not None else 0
            return int(base_len + app_len)
        return cm.local_k.size(-2)

    def get_usable_length(self, new_seq_len: int, layer_idx: int) -> int:
        # During prefill, no prior cache; during decode, return current cache KV length for the layer
        cm = self.layer_caches[layer_idx]
        if cm is None:
            return 0
        if getattr(cm, 'active_mode', False):
            base_len = cm.active_base_k.size(-2) if cm.active_base_k is not None else 0
            app_len = cm.appended_k.size(-2) if getattr(cm, 'appended_k', None) is not None else 0
            return int(base_len + app_len)

        
        return cm.local_k.size(-2)

    def _ensure_layer(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Dict[str, Any]):
        if self.layer_caches[layer_idx] is not None:
            return
        n_local = int(self.rekv_config.get('n_local', 0) or 0)
        n_init = int(self.rekv_config.get('n_init', 0) or 0)
        topk = int(self.rekv_config.get('topk', 0) or 0)
        chunk_size = int(self.rekv_config.get('chunk_size', 1) or 1)
        block_size = int(self.rekv_config.get('block_size', n_local if n_local > 0 else key_states.size(-2)))
        max_cached_block = int(self.rekv_config.get('max_cached_block', 128))
        pin_memory = bool(self.rekv_config.get('pin_memory', True))
        use_hybrid_similarity = bool(self.rekv_config.get('use_hybrid_similarity', True))
        cm = DecoupledKVManager(
            n_init=n_init,
            n_local=n_local,
            block_size=block_size,
            max_cached_block=max_cached_block,
            topk=topk,
            chunk_size=chunk_size,
            async_global_stream=True,
            pin_memory=pin_memory,
            use_hybrid_similarity=use_hybrid_similarity,
        )
        self.layer_caches[layer_idx] = cm

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Dict[str, Any]):
        """
        Called by attention forward. We ingest new KV, optionally perform retrieval if layer is in retrieval mode,
        and return (k, v) to be used this step.
        key_states/value_states: (B, n_kv_heads, L, Dh)
        Returns tensors with same dtype/device.
        """
        self._ensure_layer(layer_idx, key_states, value_states, cache_kwargs)
        cm = self.layer_caches[layer_idx]
        cm.ingest_kv(key_states, value_states)
        
        # Save query_states if flag is set and not in retrieval mode
        if getattr(cm, 'save_query_states_flag', False) and not getattr(cm, 'to_retrieve', False):
            query_states = cache_kwargs.get('query_states')
            if query_states is not None:
                cm.saved_query_states = query_states.clone().detach()
        
        if getattr(cm, 'to_retrieve', False):
            query_states = cache_kwargs.get('query_states')
            # Use saved query_states if flag is set and saved query_states exists
            if getattr(cm, 'use_saved_query_states_flag', False) and cm.saved_query_states is not None:
                query_states = cm.saved_query_states
            k_out, v_out = cm.get_retrieved_kv(query_states)
            # Persist retrieved KV as active base for subsequent growth
            cm.activate_base(k_out, v_out)
            # cat k_out and v_out with key_states and value_states
            k_out = torch.cat([k_out, key_states], dim=-2)
            v_out = torch.cat([v_out, value_states], dim=-2)
        else:
            k_out, v_out = cm.get_sliding_window_kv()
        self._seq_len += key_states.size(-2)
        return k_out, v_out

