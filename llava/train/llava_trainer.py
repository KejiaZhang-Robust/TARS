import os
import math
import random
import torch
import deepspeed
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import Optional, Dict, List, Union, Tuple
from llava.train.diff_lib import get_diff_ids

import clip

def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out

def concate_pad_three(tensorA, tensorB, tensorC, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB) + list(tensorC),
        batch_first=True,
        padding_value=padding_value)
    return out

def concate_pad_five(tensorA, tensorB, tensorC, tensorD, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB) + list(tensorC) + list(tensorD),
        batch_first=True,
        padding_value=padding_value)
    return out

def concate_pad_six(tensorA, tensorB, tensorC, tensorD, tensorE, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB) + list(tensorC) + list(tensorD) + list(tensorE),
        batch_first=True,
        padding_value=padding_value)
    return out

def concate_pad_seven(tensorA, tensorB, tensorC, tensorD, tensorE, tensorF, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB) + list(tensorC) + list(tensorD) + list(tensorE)+list(tensorF),
        batch_first=True,
        padding_value=padding_value)
    return out

def concate_pad_three_alignment_two(tensorA, tensorB, tensorC, padding_value):
    # Concatenate tensors and pad them
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB) + list(tensorC),
        batch_first=True,
        padding_value=padding_value)
    
    # Check if the tensor is 3D and if not, adjust accordingly
    if out.dim() == 3:
        # Remove the last element along the first dimension (i.e., tensor D)
        out = out[:-1, :, :]  # Remove the last batch dimension
    elif out.dim() == 2:
        # If it's 2D, we just remove the last row
        out = out[:-1, :]
    else:
        raise ValueError(f"Expected tensor with 2 or 3 dimensions, but got {out.dim()} dimensions.")
    
    return out

def concate_pad_four_alignment_three(tensorA, tensorB, tensorC, tensorD, padding_value):
    # Concatenate tensors and pad them
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB) + list(tensorC) + list(tensorD),
        batch_first=True,
        padding_value=padding_value)
    
    # Remove the last element along the first dimension (batch dimension)
    if out.dim() == 3:
        out = out[:-1, :, :]  # Remove the last batch element
    elif out.dim() == 2:
        out = out[:-1, :]  # Remove the last row
    else:
        raise ValueError(f"Expected tensor with 2 or 3 dimensions, but got {out.dim()} dimensions.")
    
    return out

def concate_pad_four_alignment_two(tensorA, tensorB, tensorC, tensorD, padding_value):
    # Concatenate tensors and pad them
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB) + list(tensorC) + list(tensorD),
        batch_first=True,
        padding_value=padding_value)
    
    # Remove the last two elements along the first dimension (batch dimension)
    if out.dim() == 3:
        out = out[:-2, :, :]  # Remove the last two batch elements
    elif out.dim() == 2:
        out = out[:-2, :]  # Remove the last two rows
    else:
        raise ValueError(f"Expected tensor with 2 or 3 dimensions, but got {out.dim()} dimensions.")
    
    return out

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        
            if torch.distributed.get_rank() == 0:
                # print(f'LR schduler is ', str(self.scheduler))
                print(f'optimizer: ', str(self.optimizer))
                print('optimizer_cls: ', optimizer_cls)
                print('optimizer_kwargs: ', optimizer_kwargs)
                print('accelerator.state: ', self.accelerator.state)
                print('self.is_deepspeed_enabled:', self.is_deepspeed_enabled)
                print('self.is_fsdp_enabled:', self.is_fsdp_enabled)

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

def chip_get_batch_logps(logits: torch.FloatTensor,
                        reference_logits: torch.FloatTensor,
                        labels: torch.LongTensor,
                        average_log_prob: bool = False):
    """Compute the kl divergence/log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
    """
    # TODO: check if sequence_length is match between them (2025-04-06 Kejia) 
    assert logits.shape[:-1] == labels.shape, (logits.shape[:-1], labels.shape)
    assert reference_logits.shape[:-1] == labels.shape, (reference_logits.shape[:-1], labels.shape)

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_policy_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1), \
                (per_policy_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
                (per_reference_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), \
                per_policy_token_logps, per_reference_token_logps
    else:
        return (per_position_kl * loss_mask).sum(-1), \
            (per_policy_token_logps * loss_mask).sum(-1), \
            (per_reference_token_logps * loss_mask).sum(-1), \
            per_policy_token_logps, per_reference_token_logps
    
def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone() # Remove the first token (by using labels[:, 1:]).
    logits = logits[:, :-1, :] # Remove the last token in logits.
    loss_mask = (labels != -100)  # Mask where labels are not -100 (ignore these).

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_policy_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_policy_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    if return_per_token_logp:
        return per_policy_token_logps

    if return_all:
        return per_policy_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob

def get_batch_logps_clean(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels[:-1].shape

    labels = labels[:-1].clone()
    labels = labels[:, 1:]     # Remove the first token (by using labels[:, 1:]).
    logits = logits[:, :-1, :] # Remove the last token in logits.
    loss_mask = (labels != -100)  # Mask where labels are not -100 (ignore these).

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_policy_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_policy_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    if return_per_token_logp:
        return per_policy_token_logps

    if return_all:
        return per_policy_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob

def compute_split_kl_alignment_loss(perturbed_logits, clean_logits, labels, win_size, rej_size, temperature=1.0):
    """
        Computes a contrastive alignment loss between perturbed and clean representations.
    """    
    labels_safe = labels[:-1].clone()
    labels_safe = labels_safe[:, 1:]     # Remove the first token (by using labels[:, 1:]).
    perturbed_logits = perturbed_logits[:, :-1, :] # Remove the last token in logits.
    clean_logits = clean_logits[:, :-1, :] # Remove the last token in logits.
    
    labels_safe[labels_safe == -100] = 0  # or any valid token id
    
    vocab_logps = perturbed_logits[:-1].log_softmax(-1)
    
    reference_vocab_ps = clean_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()
    
    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_policy_token_logps = torch.gather(vocab_logps, dim=2, index=labels_safe.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels_safe.unsqueeze(2)).squeeze(2)
    
    return (per_position_kl * labels_safe).sum(-1), \
            (per_policy_token_logps * labels_safe).sum(-1), \
            (per_reference_token_logps * labels_safe).sum(-1), \
            per_policy_token_logps, per_reference_token_logps

def get_eval_ds_config(offload=None, stage=3):
    from accelerate.state import AcceleratorState

    deepspeed_states = AcceleratorState().deepspeed_plugin

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        }
    }
    return {
        "train_micro_batch_size_per_gpu": deepspeed_states.deepspeed_config['train_micro_batch_size_per_gpu'],
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "bf16": {
            "enabled": True
        },
        "gradient_clipping": 20.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
    
def randomly_replace_tokens_with_labels(input_ids: torch.Tensor, labels: torch.Tensor, tokenizer, replace_prob=0.1, ignore_tokens=[1, 2, 0]):
    input_ids = input_ids.clone()
    labels = labels.clone()
    vocab_size = tokenizer.vocab_size
    mask = torch.rand_like(input_ids.float()) < replace_prob # 小于prob的会被扰动

    for token_id in ignore_tokens:
        mask &= (input_ids != token_id)

    random_tokens = torch.randint(low=5, high=vocab_size, size=input_ids.shape, device=input_ids.device)
    input_ids[mask] = random_tokens[mask]

    # 若某 token 被扰动，其对应 label 应该被设为 IGNORE_INDEX（不计 loss）-> 防止影响目标对齐
    ignore_index = -100
    labels[mask] = ignore_index

    return input_ids, labels

def get_token_image_similarity_scores_batched(input_ids, image, tokenizer, clip_encoder, clip_processor, device="cuda"):
    # Step 1: decode tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids)  # list[str]
    token_texts = [tokenizer.convert_tokens_to_string([t]) for t in tokens]

    # 2. Tokenize all text using clip.tokenize
    clip_texts = clip.tokenize(token_texts).to(device)  # [T, 77]

    # 3. Encode image
    image_input = clip_processor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_encoder.encode_image(image_input)  # [1, D]
        text_features = clip_encoder.encode_text(clip_texts)     # [T, D]

    ## 4. Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 5. Compute similarity via softmax
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)  # [1, T]
    similarity = similarity.squeeze(0)  # [T]

    # 6. Sort tokens by similarity
    sorted_tokens = sorted(zip(tokens, similarity.tolist()), key=lambda x: x[1])  # low → high
    
    return sorted_tokens  # list of (token, similarity score)

def generate_probabilistic_topk_mask(scores: torch.Tensor, target_ratio: float) -> torch.Tensor:
    """
    每个 token 根据 score 进行采样，但最终严格控制扰动 token 数量。
    """
    B, L = scores.shape
    total_tokens = B * L
    num_to_select = int(total_tokens * target_ratio)
    num_to_select = max(num_to_select, 1)  # 至少选择一个 token

    probs = scores.flatten()
    probs = probs / probs.sum()  # softmax 可选

    indices = torch.multinomial(probs, num_samples=num_to_select, replacement=False)

    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask[indices] = True
    return mask.view(B, L)

def perturb_tokens_with_labels(
    input_ids: torch.Tensor,                    # [B, L]
    labels: torch.Tensor,                       # [B, L]
    token_similarities: Union[List[List[float]], torch.Tensor],  # [B, L]
    tokenizer,
    replace_prob: float = 0.1,
    mode: str = "replace",                      # "replace", "swap", "mask"
    ignore_tokens: list = None,
    mask_token_id: int = None,
    ignore_index: int = -100
):
    input_ids = input_ids.clone()
    labels = labels.clone()
    B, L = input_ids.shape
    vocab_size = tokenizer.vocab_size

    # === 处理 token_similarities ===
    if isinstance(token_similarities, list):
        token_similarities = [
            [1.0 if x is None else x for x in row]
            for row in token_similarities
        ]
        sim_tensor = torch.tensor(token_similarities, device=input_ids.device)
    elif isinstance(token_similarities, torch.Tensor):
        sim_tensor = token_similarities.to(device=input_ids.device)
    else:
        raise TypeError("token_similarities must be a List[List[float]] or torch.Tensor")

    # === 截断或补齐 token_similarities ===
    if sim_tensor.shape != (B, L):
        print(f"[WARN] similarity shape {sim_tensor.shape} != input shape {(B, L)} — will fix")
        sim_tensor = F.pad(sim_tensor, (0, L - sim_tensor.shape[1]), value=1.0)[:B, :L]
    
    sim_tensor = torch.nan_to_num(sim_tensor, nan=1.0)

    # === 生成扰动概率 ===
    perturb_scores = 1.0 - sim_tensor
    max_val = perturb_scores.max()
    if max_val > 0:
        perturb_scores /= max_val
        
    # === 计算每个样本的最大值与第二大值之间的差值 ===
    topk_values, _ = torch.topk(perturb_scores, k=2, dim=1, largest=True, sorted=True)
    delta_p_perturb = topk_values[:, 0] - topk_values[:, 1]  # [B]
    delta_p_perturb = 1/(delta_p_perturb+0.01)
    delta_p_perturb = delta_p_perturb*0.001
    # print(delta_p_perturb)
    delta_p_perturb = min(delta_p_perturb, 0.50)
    #TODO： new
    # mask = generate_probabilistic_topk_mask(perturb_scores, target_ratio=replace_prob)

    mask = generate_probabilistic_topk_mask(perturb_scores, target_ratio=delta_p_perturb)
    
    # === 忽略特殊 token（比如 pad、bos、eos）
    if ignore_tokens is None:
        ignore_tokens = [
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        ]
    for token_id in ignore_tokens:
        mask &= (input_ids != token_id)

    # === 扰动逻辑 ===
    if mode == "replace":
        random_tokens = torch.randint(low=5, high=vocab_size, size=(B, L), device=input_ids.device)
        input_ids[mask] = random_tokens[mask]

    elif mode == "swap":
        for b in range(B):
            positions = torch.where(mask[b])[0]
            for i in range(0, len(positions) - 1, 2):
                l1, l2 = positions[i], positions[i + 1]
                input_ids[b, l1], input_ids[b, l2] = input_ids[b, l2], input_ids[b, l1]

    elif mode == "mask":
        if mask_token_id is None:
            mask_token_id = tokenizer.mask_token_id or tokenizer.unk_token_id or tokenizer.pad_token_id or 0
        input_ids[mask] = mask_token_id

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    labels[mask] = ignore_index
    return input_ids, labels, mask

def perturb_tokens_randomly(
    input_ids: torch.Tensor,                    # [B, L]
    labels: torch.Tensor,                       # [B, L]
    token_similarities: Union[List[List[float]], torch.Tensor],  # ignored here
    tokenizer,
    replace_prob: float = 0.1,
    mode: str = "replace",                      # "replace", "swap", "mask"
    ignore_tokens: list = None,
    mask_token_id: int = None,
    ignore_index: int = -100
):
    input_ids = input_ids.clone()
    labels = labels.clone()
    B, L = input_ids.shape
    vocab_size = tokenizer.vocab_size

    # === 默认忽略特殊 token（pad、bos、eos）===
    if ignore_tokens is None:
        ignore_tokens = [
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        ]

    # === 随机生成 mask ===
    rand = torch.rand((B, L), device=input_ids.device)
    mask = rand < replace_prob

    for token_id in ignore_tokens:
        mask &= (input_ids != token_id)

    # === 扰动逻辑 ===
    if mode == "replace":
        random_tokens = torch.randint(low=5, high=vocab_size, size=(B, L), device=input_ids.device)
        input_ids[mask] = random_tokens[mask]

    elif mode == "swap":
        for b in range(B):
            positions = torch.where(mask[b])[0]
            for i in range(0, len(positions) - 1, 2):
                l1, l2 = positions[i], positions[i + 1]
                input_ids[b, l1], input_ids[b, l2] = input_ids[b, l2], input_ids[b, l1]

    elif mode == "mask":
        if mask_token_id is None:
            mask_token_id = tokenizer.mask_token_id or tokenizer.unk_token_id or tokenizer.pad_token_id or 0
        input_ids[mask] = mask_token_id

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    labels[mask] = ignore_index
    return input_ids, labels, mask

class LLAVADPOTrainer(LLaVATrainer):
    def __init__(self, ref_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        if torch.distributed.get_rank() == 0:
            print('self.args:', self.args)
        if self.ref_model is not None and 'zero3' in self.args.deepspeed:
            eval_ds_config = get_eval_ds_config(offload=False)
            self.ref_model, *_ = deepspeed.initialize(model=self.ref_model, config=eval_ds_config)
            self.ref_model.eval()
            print('ref_model deepspeed init done!')

    def chip_loss(self, policy_chosen_logp: torch.FloatTensor,
                    policy_rejected_logp: torch.FloatTensor,
                    policy_win_diffusionImage_logp: torch.FloatTensor,
                    reference_chosen_logp: torch.FloatTensor,
                    reference_rejected_logp: torch.FloatTensor,
                    uncond_ref_win_logp: torch.FloatTensor,
                    uncond_ref_rej_logp: torch.FloatTensor,
                    chosen_position_kl: torch.FloatTensor,
                    rejected_position_kl: torch.FloatTensor,
                    beta: float=0.1, gama:float=0.3
                    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the TDPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logp: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logp: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            policy_win_diffusionImage_logp: Log probabilities of the policy model for the chosen responses with diffusion image. Shape: (batch_size,)
            reference_chosen_logp: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logp: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            uncond_ref_win_logp: unconditional Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            uncond_ref_rej_logp: unconditional Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            chosen_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            rejected_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss
            gama: Temperature parameter for the CMDPO loss

        Returns:
            A tuple of two tensors: (losses, rewards).
            The losses tensor contains the TDPO loss for each example in the batch.
            The rewards tensors contain the rewards for response pair.
        """
        pi_logratios = policy_chosen_logp - policy_rejected_logp
        ref_logratios = reference_chosen_logp - reference_rejected_logp
        logits = pi_logratios - ref_logratios
        if self.args.use_cross_modal_loss:
            logits += policy_chosen_logp - reference_chosen_logp
            logits -= policy_win_diffusionImage_logp - uncond_ref_win_logp        
        if self.args.use_tdpo:
            logits -= self.args.tok_beta * (
                         rejected_position_kl - chosen_position_kl.detach())
            chosen_values = policy_chosen_logp - reference_chosen_logp + chosen_position_kl
            rejected_values = policy_rejected_logp - reference_rejected_logp + rejected_position_kl
        else:
            chosen_values = policy_chosen_logp - reference_chosen_logp
            rejected_values = policy_rejected_logp - reference_rejected_logp
        
        losses = -F.logsigmoid(beta * logits)

        chosen_rewards = beta * chosen_values.detach()
        rejected_rewards = beta * rejected_values.detach()

        return losses, chosen_rewards, rejected_rewards

    def adv_loss(self, policy_chosen_logp: torch.FloatTensor,
                    policy_rejected_logp: torch.FloatTensor,
                    policy_win_diffusionImage_logp: torch.FloatTensor,
                    uncond_ref_win_logp: torch.FloatTensor,
                    policy_rej_diffusionImage_logp: torch.FloatTensor,
                    uncond_ref_rej_logp: torch.FloatTensor,
                    clean_policy_chosen_logp: torch.FloatTensor,
                    clean_policy_rejected_logp: torch.FloatTensor,
                    reference_chosen_logp: torch.FloatTensor,
                    reference_rejected_logp: torch.FloatTensor,
                    clean_win_logp: torch.FloatTensor,
                    clean_rej_logp: torch.FloatTensor,
                    chosen_position_kl_adv: torch.FloatTensor,
                    rejected_position_kl_adv: torch.FloatTensor,
                    clean_chosen_position_kl: torch.FloatTensor, 
                    clean_rejected_position_kl: torch.FloatTensor,
                    beta: float=0.1, gama:float=0.3
                    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        pi_logratios = policy_chosen_logp - policy_rejected_logp
        ref_logratios = reference_chosen_logp - reference_rejected_logp
        logits = pi_logratios - ref_logratios
        
        logits_negtive = policy_chosen_logp - reference_chosen_logp - (policy_win_diffusionImage_logp - uncond_ref_win_logp)
        
        losses = -F.logsigmoid(beta * logits) - F.logsigmoid(beta * logits_negtive)
        
        chosen_values = policy_chosen_logp - reference_chosen_logp
        rejected_values = policy_rejected_logp - reference_rejected_logp

        chosen_rewards = beta * chosen_values.detach()
        rejected_rewards = beta * rejected_values.detach()

        return losses, chosen_rewards, rejected_rewards

    def compute_loss(self, model: Module, inputs: dict, tokenizer=None, return_outputs=False):
        data_dict = inputs
        win_input_ids = data_dict.pop('win_input_ids')
        rej_input_ids = data_dict.pop('rej_input_ids')
        win_labels = data_dict.pop('win_labels')
        rej_labels = data_dict.pop('rej_labels')
        pad_token_id = data_dict.pop('pad_token_id')
        
        win_input_token_similarities=data_dict.pop('win_input_token_similarities')
        rej_input_token_similarities=data_dict.pop('rej_input_token_similarities')

        images = data_dict.pop('images')
        diffusion_image = data_dict.pop('diffusion_image', '')
        win_size = win_input_ids.shape[0]
        rej_size = rej_input_ids.shape[0]
        assert win_size == rej_size

        if self.args.use_tokenizer_adversarial:
            original_win_input_ids = win_input_ids.clone()
            original_rej_input_ids = rej_input_ids.clone()

            win_input_ids, win_labels, win_mask = perturb_tokens_with_labels(
                win_input_ids, win_labels, win_input_token_similarities, mode=self.args.token_mode_adv, tokenizer=tokenizer, replace_prob=self.args.adv_p)
            rej_input_ids, rej_labels, rej_mask = perturb_tokens_with_labels(
                rej_input_ids, rej_labels, rej_input_token_similarities, mode=self.args.token_mode_adv, tokenizer=tokenizer, replace_prob=self.args.adv_p)
             
        concatenated_input_ids_6 = concate_pad_seven(win_input_ids, rej_input_ids, original_win_input_ids, original_rej_input_ids, win_input_ids, rej_input_ids, pad_token_id)
        concatenated_labels_6 = concate_pad_seven(win_labels, rej_labels, win_labels, rej_labels, win_labels, rej_labels, -100)
        concatenated_attention_mask_6 = concatenated_input_ids_6.ne(pad_token_id)
        
        ref_logps = data_dict.pop('offline_ref_logits', None)
        if ref_logps is not None:
            ref_logps = torch.as_tensor(ref_logps).cuda()
        idx = data_dict.pop('idx', None)

        output, new_labels = model(
            input_ids=concatenated_input_ids_6,
            labels=concatenated_labels_6,
            attention_mask=concatenated_attention_mask_6,
            images=torch.cat([images, images, images, images, diffusion_image, diffusion_image], dim=0),
            return_new_labels=True,
            output_hidden_states=True,
            **data_dict
        )

        with torch.no_grad():
            ref_output = self.ref_model(
                input_ids=concatenated_input_ids_6,
                labels=concatenated_labels_6,
                attention_mask=concatenated_attention_mask_6,
                images=torch.cat([images, images, images, images, diffusion_image, diffusion_image], dim=0),
                output_hidden_states=True,
                **data_dict
            )
        
        all_position_kl, policy_logps, ref_logps, \
            per_policy_token_logps, per_reference_token_logps = chip_get_batch_logps(
            output.logits, ref_output.logits,
            new_labels, average_log_prob=False)
        
        chosen_position_kl, rejected_position_kl, clean_chosen_position_kl, clean_rejected_position_kl, policy_diffusion_win_kl, policy_diffusion_rej_kl  = all_position_kl.split([win_size, rej_size, win_size, rej_size, win_size, rej_size])
        
        # three-tuple logits
        reference_chosen_logp, reference_rejected_logp, clean_reference_chosen_logp, clean_reference_rejected_logp, reference_diffusion_chosen_logp, reference_diffusion_rejected_logp = ref_logps.split([win_size, rej_size, win_size, rej_size, win_size, rej_size])

        policy_chosen_logp, policy_rejected_logp, clean_policy_chosen_logp, clean_policy_rejected_logp, policy_diffusion_shosen_logp, policy_diffusion_rejected_logp = policy_logps.split([win_size, rej_size, win_size, rej_size, win_size, rej_size])

        all_position_kl_adv_clean, _, _, \
            _, _ = chip_get_batch_logps(
            output.logits[:2], ref_output.logits[-2:],
            new_labels[:2], average_log_prob=False)
            
        win_kl_adv_clean, rej_kl_adv_clean = all_position_kl_adv_clean.split([win_size, rej_size])
        
        losses, chosen_rewards, rejected_rewards = self.adv_loss(
            policy_chosen_logp, policy_rejected_logp, 
            policy_diffusion_shosen_logp, reference_diffusion_chosen_logp,
            policy_diffusion_rejected_logp, reference_diffusion_rejected_logp,
            clean_policy_chosen_logp, clean_policy_rejected_logp,
            reference_chosen_logp, reference_rejected_logp,
            clean_reference_chosen_logp, clean_reference_rejected_logp,
            chosen_position_kl, rejected_position_kl,
            win_kl_adv_clean, rej_kl_adv_clean)

        def frequency_triplet_loss(z_pw, z_cw, z_cr, margin=0.5):
            def to_fft(x):
                x = x.to(torch.float32)
                fft = torch.fft.fft(x, dim=0)
                return torch.abs(fft)        

            z_pw_f = to_fft(z_pw)
            z_cw_f = to_fft(z_cw)
            z_cr_f = to_fft(z_cr)

            pos_dist = F.mse_loss(z_pw_f, z_cw_f, reduction='mean')     # z_pw vs z_cw
            neg_dist = F.mse_loss(z_pw_f, z_cr_f, reduction='mean')     # z_pw vs z_cr

            loss = F.relu(pos_dist - neg_dist + margin)

            return loss

        output_hidden_emd = output.hidden_states[0] # [perturbed_win, perturbed_rej, clean_win, clean_rej]
        ref_hidden_emd = ref_output.hidden_states[0] # [perturbed_win, perturbed_rej, clean_win, clean_rej]

        z_pw = output_hidden_emd[0]  # perturbed_win
        z_cw = ref_hidden_emd[2]     # clean_win
        z_cr = ref_hidden_emd[3]     # clean_rej

        freq_align_loss = frequency_triplet_loss(z_pw, z_cw, z_cr, margin=0.5)
            
        loss = losses.mean() + self.args.beta_fre * freq_align_loss
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        train_test = 'train' if model.training else 'test'
        metrics = {}
        metrics[f'rewards_{train_test}/chosen'] = self._nested_gather(chosen_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/rejected'] = self._nested_gather(rejected_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/accuracies'] = self._nested_gather(reward_accuracies.mean()).mean().item()
        metrics[f'rewards_{train_test}/margins'] = metrics[f'rewards_{train_test}/chosen'] - metrics[f'rewards_{train_test}/rejected']
        metrics[f'logps_{train_test}/rejected'] = self._nested_gather(policy_rejected_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/chosen'] = self._nested_gather(policy_chosen_logp.mean()).mean().item()
        metrics['loss'] = float(loss)
        self.log(metrics)
        return loss

    def get_seg_weight(self, 
                        win_labels, rej_labels,
                        win_input_ids, rej_input_ids
                        ):
        win_token_weight = torch.ones_like(win_labels[:, 1:], dtype=torch.bfloat16)
        rej_token_weight = torch.ones_like(rej_labels[:, 1:], dtype=torch.bfloat16)
        for idx, (w, r) in enumerate(zip(win_input_ids, rej_input_ids)):
            valid_w = w[1:]
            valid_r = r[1:]
            min_match_size = 3
            r_mod, w_mod = get_diff_ids(valid_r.tolist(), valid_w.tolist(), min_match_size=min_match_size)
            win_token_weight[idx][w_mod] = self.args.dpo_token_weight
            rej_token_weight[idx][r_mod] = self.args.dpo_token_weight

        return win_token_weight, rej_token_weight
    
    @staticmethod    
    def compute_weighted_logp(per_token_logp, labels, token_weight, use_average=False):
        loss_mask = (labels[:, 1:].clone() != -100)
        weighted_mask = token_weight * loss_mask
        if len(per_token_logp.shape)!=1:
            per_token_logp = per_token_logp[:, -weighted_mask.shape[1]:]
        logp = (per_token_logp * weighted_mask).sum(-1)

        average_logp = logp / weighted_mask.sum(-1)
        if use_average:
            return average_logp

        return logp
