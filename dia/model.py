import time
from enum import Enum

import dac
import numpy as np
import torch
import torchaudio

# Assuming these imports are relative to the package structure
from .audio import apply_audio_delay, build_delay_indices, build_revert_indices, decode, revert_audio_delay
from .config import DiaConfig
from .layers import DiaModel
from .state import DecoderInferenceState, DecoderOutput, EncoderInferenceState


DEFAULT_SAMPLE_RATE = 44100
SAMPLE_RATE_RATIO = 512


def _get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sample_next_token(
    logits_BCxV: torch.Tensor,
    temperature: float,
    top_p: float,
    audio_eos_value: int,
) -> torch.Tensor:
    if temperature == 0.0:
        return torch.argmax(logits_BCxV, dim=-1)

    logits_BCxV = logits_BCxV / temperature

    if audio_eos_value is not None and audio_eos_value >= 0:
        top_logit_indices_BC = torch.argmax(logits_BCxV, dim=-1)
        eos_not_highest_mask_BC = top_logit_indices_BC != audio_eos_value
        mask_eos_unless_highest_BCxV = torch.zeros_like(logits_BCxV, dtype=torch.bool)
        if eos_not_highest_mask_BC.any():
            mask_eos_unless_highest_BCxV[eos_not_highest_mask_BC, audio_eos_value] = True
        logits_BCxV = logits_BCxV.masked_fill(mask_eos_unless_highest_BCxV, -torch.inf)

    if top_p < 1.0:
        probs_BCxV = torch.softmax(logits_BCxV, dim=-1)
        sorted_probs_BCxV, sorted_indices_BCxV = torch.sort(probs_BCxV, dim=-1, descending=True)
        cumulative_probs_BCxV = torch.cumsum(sorted_probs_BCxV, dim=-1)

        sorted_indices_to_remove_BCxV = cumulative_probs_BCxV > top_p
        sorted_indices_to_remove_BCxV[..., 1:] = sorted_indices_to_remove_BCxV[..., :-1].clone()
        sorted_indices_to_remove_BCxV[..., 0] = 0

        indices_to_remove_BCxV = torch.zeros_like(sorted_indices_to_remove_BCxV)
        indices_to_remove_BCxV.scatter_(dim=-1, index=sorted_indices_BCxV, src=sorted_indices_to_remove_BCxV)
        logits_BCxV = logits_BCxV.masked_fill(indices_to_remove_BCxV, -torch.inf)

    final_probs_BCxV = torch.softmax(logits_BCxV, dim=-1)

    sampled_indices_BC = torch.multinomial(final_probs_BCxV, num_samples=1)
    sampled_indices_C = sampled_indices_BC.squeeze(-1)
    return sampled_indices_C


class ComputeDtype(str, Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"

    def to_dtype(self) -> torch.dtype:
        if self == ComputeDtype.FLOAT32:
            return torch.float32
        elif self == ComputeDtype.FLOAT16:
            return torch.float16
        elif self == ComputeDtype.BFLOAT16:
            return torch.bfloat16
        else:
            raise ValueError(f"Unsupported compute dtype: {self}")


class Dia:
    def __init__(
        self,
        config: DiaConfig,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
    ):
        """Initializes the Dia model.

        Args:
            config: The configuration object for the model.
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.

        Raises:
            RuntimeError: If there is an error loading the DAC model.
        """
        super().__init__()
        self.config = config
        self.device = device if device is not None else _get_default_device()
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)
        self.compute_dtype = compute_dtype.to_dtype()
        self.model = DiaModel(config, self.compute_dtype)
        self.dac_model = None
        self._compiled_step = None

    @classmethod
    def from_local(
        cls,
        config_path: str,
        checkpoint_path: str,
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
    ) -> "Dia":
        """Loads the Dia model from local configuration and checkpoint files.

        Args:
            config_path: Path to the configuration JSON file.
            checkpoint_path: Path to the model checkpoint (.pth) file.
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If the config or checkpoint file is not found.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        dia = cls(config, compute_dtype, device)

        try:
            state_dict = torch.load(checkpoint_path, map_location=dia.device)
            dia.model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading checkpoint from {checkpoint_path}") from e

        dia.model.to(dia.device)
        dia.model.eval()
        dia._load_dac_model()
        return dia

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "nari-labs/Dia-1.6B",
        compute_dtype: str | ComputeDtype = ComputeDtype.FLOAT32,
        device: torch.device | None = None,
    ) -> "Dia":
        """Loads the Dia model from a Hugging Face Hub repository.

        Downloads the configuration and checkpoint files from the specified
        repository ID and then loads the model.

        Args:
            model_name: The Hugging Face Hub repository ID (e.g., "nari-labs/Dia-1.6B").
            compute_dtype: The computation dtype to use.
            device: The device to load the model onto. If None, will automatically select the best available device.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If config or checkpoint download/loading fails.
            RuntimeError: If there is an error loading the checkpoint.
        """
        if isinstance(compute_dtype, str):
            compute_dtype = ComputeDtype(compute_dtype)

        # Load model directly using DiaModel's from_pretrained which handles HF download
        try:
            loaded_model = DiaModel.from_pretrained(model_name, compute_dtype=compute_dtype.to_dtype())
        except Exception as e:
            raise RuntimeError(f"Error loading model from Hugging Face Hub ({model_name})") from e

        config = loaded_model.config  # Get config from the loaded model
        dia = cls(config, compute_dtype, device)

        dia.model = loaded_model  # Assign the already loaded model
        dia.model.to(dia.device)
        dia.model.eval()
        dia._load_dac_model()
        return dia

    def _load_dac_model(self):
        try:
            dac_model_path = dac.utils.download()
            dac_model = dac.DAC.load(dac_model_path).to(self.device)
            dac_model.eval()  # Ensure DAC is in eval mode
        except Exception as e:
            raise RuntimeError("Failed to load DAC model") from e
        self.dac_model = dac_model

    def _prepare_text_input(self, text: str) -> torch.Tensor:
        """Encodes text prompt, pads, and creates attention mask and positions."""
        text_pad_value = self.config.data.text_pad_value
        max_len = self.config.data.text_length

        byte_text = text.encode("utf-8")
        # Replace special tokens with their byte values if needed by the specific tokenizer/config
        # Assuming byte values 1 and 2 are correct placeholders based on original code
        replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
        text_tokens = list(replaced_bytes)  # Convert to list of integer byte values

        current_len = len(text_tokens)
        padding_needed = max_len - current_len
        if padding_needed <= 0:
            # Truncate if too long
            text_tokens = text_tokens[:max_len]
            padded_text_np = np.array(text_tokens, dtype=np.uint8)
        else:
            # Pad if too short
            padded_text_np = np.pad(
                text_tokens,
                (0, padding_needed),
                mode="constant",
                constant_values=text_pad_value,
            ).astype(np.uint8)

        # Convert to tensor
        src_tokens = torch.from_numpy(padded_text_np).to(torch.long).to(self.device).unsqueeze(0)  # [1, S] (use long)
        return src_tokens

    def _prepare_audio_prompt(self, audio_prompts: list[torch.Tensor | None]) -> tuple[torch.Tensor, list[int]]:
        num_channels = self.config.data.channels
        audio_bos_value = self.config.data.audio_bos_value
        delay_pattern = self.config.data.delay_pattern
        max_delay_pattern = max(delay_pattern)
        batch_size = len(audio_prompts)

        prepared_prompts = []
        prefill_steps = []

        bos = torch.full(
            (1, num_channels),
            fill_value=audio_bos_value,
            dtype=torch.int,
            device=self.device,
        )

        for i in range(batch_size):
            prompt = audio_prompts[i]

            if prompt is None:
                current_prompt = bos
            else:
                # Ensure prompt is on the correct device and has correct dtype
                prompt = prompt.to(device=self.device, dtype=torch.long)  # Use long
                if prompt.dim() != 2 or prompt.shape[1] != num_channels:
                    raise ValueError(
                        f"Audio prompt {i} has incorrect shape {prompt.shape}. Expected [T, {num_channels}]"
                    )
                current_prompt = torch.cat([bos, prompt], dim=0)

            current_len = current_prompt.shape[0]
            prepared_prompts.append(current_prompt)
            prefill_steps.append(current_len)

        # Pad each prompt to the max length in the batch
        max_len = max(p.shape[0] for p in prepared_prompts)
        padded_prompts = []
        for p in prepared_prompts:
            padding_needed = max_len - p.shape[0]
            if padding_needed > 0:
                pad_tensor = torch.full(
                    (padding_needed, num_channels),
                    fill_value=-1,  # Use -1 as temporary pad before delay application
                    dtype=torch.int,
                    device=self.device,
                )
                p = torch.cat([p, pad_tensor], dim=0)
            padded_prompts.append(p)

        # Stack into a batch
        batched_prompts = torch.stack(padded_prompts, dim=0)  # Shape [B, max_len, C]

        # Add delay pattern padding
        delay_pad_tensor = torch.full(
            (batch_size, max_delay_pattern, num_channels),
            fill_value=-1,  # Use -1 for delay padding before apply_audio_delay
            dtype=torch.int,
            device=self.device,
        )
        # Shape [B, max_len + max_delay_pattern, C]
        batched_prompts_padded = torch.cat([batched_prompts, delay_pad_tensor], dim=1)

        # Apply delay pattern
        T_padded = batched_prompts_padded.shape[1]
        delay_precomp = build_delay_indices(
            B=batch_size,
            T=T_padded,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        # Apply audio delay expects long tensor and replaces -1 with pad_value
        delayed_batch = apply_audio_delay(
            audio_BxTxC=batched_prompts_padded,
            pad_value=-1,
            bos_value=audio_bos_value,
            precomp=delay_precomp,
        )

        return delayed_batch, prefill_steps

    def _prepare_generation(
        self,
        text: str | list[str],
        audio_prompts: list[str | torch.Tensor | None],
        verbose: bool,
    ):
        if isinstance(text, list):
            batch_size = len(text)
            enc_input_cond_batch = torch.cat([self._prepare_text_input(t) for t in text], dim=0)  # Shape: [B, S]
            enc_input_uncond_batch = torch.zeros_like(enc_input_cond_batch)  # Shape: [B, S]
            stacked_inputs = torch.stack([enc_input_uncond_batch, enc_input_cond_batch], dim=1)  # Shape: [B, 2, S]
            enc_input = stacked_inputs.view(2 * batch_size, -1)  # Shape: [2 * B, S]
            enc_input_cond = enc_input_cond_batch
        else:
            batch_size = 1
            enc_input_cond = self._prepare_text_input(text)
            enc_input_uncond = torch.zeros_like(enc_input_cond)
            enc_input = torch.cat([enc_input_uncond, enc_input_cond], dim=0)

        # Process audio prompts (load if paths are given)
        loaded_audio_prompts = []
        for ap in audio_prompts:
            if isinstance(ap, str):
                loaded_audio_prompts.append(self.load_audio(ap))
            else:
                loaded_audio_prompts.append(ap)  # Assumes Tensor or None

        # Get batched, delayed prefill tokens and original lengths
        prefill_batch, prefill_steps = self._prepare_audio_prompt(loaded_audio_prompts)

        if verbose:
            print("generate: data loaded")

        # Expand encoder state for CFG (double the batch size)
        # Assuming enc_state needs to match the doubled batch size for decoder
        enc_state = EncoderInferenceState.new(self.config, enc_input_cond)
        encoder_out = self.model.encoder(enc_input, enc_state)

        dec_cross_attn_cache = self.model.decoder.precompute_cross_attn_cache(
            encoder_out, enc_state.positions, enc_state.padding_mask
        )
        dec_state = DecoderInferenceState.new(
            self.config, enc_state, encoder_out, dec_cross_attn_cache, self.compute_dtype
        )
        dec_output = DecoderOutput.new(batch_size, self.config, self.device)
        dec_output.prefill(prefill_batch, prefill_steps)

        dec_step = min(prefill_steps) - 1
        if dec_step > 0:
            dec_state.prepare_step(0, dec_step)
            tokens_BxTxC = dec_output.get_tokens_at(0, dec_step).repeat_interleave(2, dim=0)
            self.model.decoder.forward(tokens_BxTxC, dec_state)

        return dec_state, dec_output

    def _decoder_step(
        self,
        tokens_Bx1xC: torch.Tensor,  # Shape [2*B, 1, C]
        dec_state: DecoderInferenceState,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        cfg_filter_top_k: int | None,  # Use this for CFG-Filter 'k'
    ) -> torch.Tensor:  # Returns shape [B, C]
        audio_eos_value = self.config.data.audio_eos_value

        logits_2Bx1xCxV = self.model.decoder.decode_step(tokens_Bx1xC, dec_state)
        B = tokens_Bx1xC.shape[0] // 2

        logits_last_2BxCxV = logits_2Bx1xCxV[:, -1, :, :]
        logits_last_Bx2xCxV = logits_last_2BxCxV.view(B, 2, *logits_last_2BxCxV.shape[1:])

        uncond_logits_BxCxV = logits_last_Bx2xCxV[:, 0, :, :]  # Shape [B, C, V]
        cond_logits_BxCxV = logits_last_Bx2xCxV[:, 1, :, :]  # Shape [B, C, V]

        cfg_logits_BxCxV = cond_logits_BxCxV + cfg_scale * (cond_logits_BxCxV - uncond_logits_BxCxV)

        if cfg_filter_top_k is not None and cfg_filter_top_k > 0:
            k = cfg_filter_top_k
            _, top_k_indices_BxCxk = torch.topk(cond_logits_BxCxV, k=k, dim=-1)
            mask_BxCxV = torch.ones_like(cfg_logits_BxCxV, dtype=torch.bool)
            mask_BxCxV.scatter_(dim=-1, index=top_k_indices_BxCxk, value=False)
            sample_logits_BxCxV = cfg_logits_BxCxV.masked_fill(mask_BxCxV, -torch.inf)
        else:
            sample_logits_BxCxV = cfg_logits_BxCxV

        if audio_eos_value is not None and audio_eos_value >= 0:
            sample_logits_BxCxV[:, :, audio_eos_value + 1 :] = -torch.inf
            sample_logits_BxCxV[:, 1:, audio_eos_value] = -torch.inf
            sample_logits_BxCxV[:, 0, audio_eos_value] *= 0.8

        flat_logits_BCxV = sample_logits_BxCxV.view(B * self.config.data.channels, -1)

        pred_BC = _sample_next_token(
            flat_logits_BCxV.float(),
            temperature=temperature,
            top_p=top_p,
            audio_eos_value=audio_eos_value,
        )

        pred_BxC = pred_BC.view(B, self.config.data.channels)
        return pred_BxC

    def _generate_output(self, generated_codes: torch.Tensor, lengths_Bx: torch.Tensor) -> list[np.ndarray]:
        num_channels = self.config.data.channels
        batch_size = generated_codes.shape[0]
        seq_length = generated_codes.shape[1]
        delay_pattern = self.config.data.delay_pattern
        audio_pad_value = self.config.data.audio_pad_value
        max_delay_pattern = max(delay_pattern)

        revert_precomp = build_revert_indices(
            B=batch_size,
            T=seq_length,
            C=num_channels,
            delay_pattern=delay_pattern,
        )

        codebook = revert_audio_delay(
            audio_BxTxC=generated_codes,
            pad_value=audio_pad_value,
            precomp=revert_precomp,
            T=seq_length,
        )[:, :-max_delay_pattern, :]

        min_valid_index = 0
        max_valid_index = 1023
        invalid_mask = (codebook < min_valid_index) | (codebook > max_valid_index)
        codebook[invalid_mask] = 0

        audios = []

        for i in range(batch_size):
            audio = decode(self.dac_model, codebook[i, : lengths_Bx[i], :].unsqueeze(0).transpose(1, 2))
            audio_np = audio.squeeze().cpu().numpy()
            audios.append(audio_np)
        return audios

    def load_audio(self, audio_path: str) -> torch.Tensor:
        audio, sr = torchaudio.load(audio_path, channels_first=True)  # C, T
        if sr != DEFAULT_SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, DEFAULT_SAMPLE_RATE)
        audio = audio.to(self.device).unsqueeze(0)  # 1, C, T
        audio_data = self.dac_model.preprocess(audio, DEFAULT_SAMPLE_RATE)
        _, encoded_frame, _, _, _ = self.dac_model.encode(audio_data)  # 1, C, T
        return encoded_frame.squeeze(0).transpose(0, 1)

    def save_audio(self, path: str, audio: np.ndarray):
        import soundfile as sf

        sf.write(path, audio, DEFAULT_SAMPLE_RATE)

    @torch.inference_mode()
    def generate(
        self,
        text: str | list[str],
        max_tokens: int | None = None,
        cfg_scale: float = 4.0,
        temperature: float = 1.5,
        top_p: float = 0.95,
        use_torch_compile: bool = False,
        cfg_filter_top_k: int = 25,
        audio_prompt: list[str | torch.Tensor | None] | str | torch.Tensor | None = None,
        audio_prompt_path: list[str | torch.Tensor | None] | str | torch.Tensor | None = None,
        use_cfg_filter: bool | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        # --- Boilerplate setup ---
        batch_size = len(text) if isinstance(text, list) else 1
        audio_eos_value = self.config.data.audio_eos_value
        audio_pad_value = self.config.data.audio_pad_value
        delay_pattern = self.config.data.delay_pattern
        max_tokens = self.config.data.audio_length if max_tokens is None else max_tokens
        max_delay_pattern = max(delay_pattern)
        delay_pattern_Cx = torch.tensor(delay_pattern, device=self.device, dtype=torch.long)
        self.model.eval()

        if audio_prompt_path:
            print("Warning: audio_prompt_path is deprecated. Use audio_prompt instead.")
            audio_prompt = audio_prompt_path
        if use_cfg_filter is not None:
            print("Warning: use_cfg_filter is deprecated.")

        if verbose:
            total_start_time = time.time()

        if not isinstance(audio_prompt, list):
            audio_prompt = [audio_prompt] * batch_size  # Keep this update

        assert len(audio_prompt) == batch_size, "Number of audio prompts must match batch size"  # Keep this update

        # --- Prepare inputs and state ---
        dec_state, dec_output = self._prepare_generation(text, audio_prompt, verbose)
        dec_step = min(dec_output.prefill_steps) - 1  # Revert to original start step logic

        # --- Batch-aware EOS tracking (NEW)---
        eos_detected_Bx = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        eos_countdown_Bx = torch.full(
            (batch_size,), -1, dtype=torch.long, device=self.device
        )  # -1:inactive, >0:countdown, 0:finished
        finished_step_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

        bos_over = False  # Original flag

        # --- Compile step function if needed ---
        if use_torch_compile:
            if self._compiled_step is None:
                step_fn = torch.compile(self._decoder_step, mode="default")
                self._compiled_step = step_fn
            else:
                step_fn = self._compiled_step
        else:
            step_fn = self._decoder_step

        if verbose:
            print("generate: starting generation loop")
            if use_torch_compile:
                print("generate: using use_torch_compile=True, the first step may be slow")  # Original message
            start_time = time.time()  # Original timing variable

        # --- Generation Loop ---
        while dec_step < max_tokens:
            current_step_idx = dec_step + 1  # Keep for clarity

            # Check if all sequences have finished counting down EOS padding (NEW)
            if (eos_countdown_Bx == 0).all():
                break

            dec_state.prepare_step(dec_step)
            tokens_Bx1xC = dec_output.get_tokens_at(dec_step).repeat_interleave(2, dim=0)  # Repeat for CFG

            pred_BxC = step_fn(
                tokens_Bx1xC,
                dec_state,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
            )  # Shape [B, C]

            # --- Batch EOS Detection & Trigger (NEW)---
            active_mask_Bx = eos_countdown_Bx != 0
            eos_trigger_Bx = torch.zeros_like(active_mask_Bx)
            if active_mask_Bx.any():
                is_eos_token = (~eos_detected_Bx[active_mask_Bx]) & (pred_BxC[active_mask_Bx, 0] == audio_eos_value)
                # Original single condition check for max length:
                # is_max_len = (dec_step == max_tokens - max_delay_pattern - 1)
                # Updated check using current_step_idx:
                is_max_len = current_step_idx >= max_tokens - max_delay_pattern
                eos_trigger_Bx[active_mask_Bx] = is_eos_token | is_max_len
            eos_detected_Bx |= eos_trigger_Bx
            start_countdown_mask_Bx = eos_trigger_Bx & (eos_countdown_Bx < 0)
            if start_countdown_mask_Bx.any():
                eos_countdown_Bx[start_countdown_mask_Bx] = max_delay_pattern
                finished_step_Bx[start_countdown_mask_Bx] = current_step_idx

            # --- Batch EOS Padding (Vectorized) (NEW) ---
            padding_mask_Bx = eos_countdown_Bx > 0
            if padding_mask_Bx.any():
                pred_active_BxC = pred_BxC[padding_mask_Bx].clone()
                countdown_active_Bx = eos_countdown_Bx[padding_mask_Bx]
                step_after_eos_Bx = max_delay_pattern - countdown_active_Bx
                step_after_eos_Bx_ = step_after_eos_Bx.unsqueeze(1)
                delay_pattern_Cx_ = delay_pattern_Cx.unsqueeze(0)
                eos_mask_NxC = step_after_eos_Bx_ == delay_pattern_Cx_
                pad_mask_NxC = step_after_eos_Bx_ > delay_pattern_Cx_
                pred_active_BxC[eos_mask_NxC] = audio_eos_value
                pred_active_BxC[pad_mask_NxC] = audio_pad_value
                pred_BxC[padding_mask_Bx] = pred_active_BxC
                eos_countdown_Bx[padding_mask_Bx] -= 1

            # --- Update BOS flag (Original) ---
            if not bos_over:
                bos_over = all(
                    dec_step - prefill_step > max_delay_pattern for prefill_step in dec_output.prefill_steps
                )

            dec_output.update_one(pred_BxC, current_step_idx, not bos_over)

            dec_step += 1

            if verbose and dec_step % 86 == 0:
                duration = time.time() - start_time
                if duration > 0:
                    print(
                        f"generate step {dec_step}: speed={86 * batch_size / duration:.3f} tokens/s, realtime factor={batch_size / duration:.3f}x"
                    )
                start_time = time.time()

        # --- Finalize and Extract Output ---
        final_step = dec_step + 1

        finished_step_Bx[finished_step_Bx == -1] = final_step - max_delay_pattern

        prefill_steps_tensor = torch.tensor(dec_output.prefill_steps, device=self.device)
        lengths_Bx = finished_step_Bx - prefill_steps_tensor
        lengths_Bx = torch.clamp(lengths_Bx, min=0)

        max_len = lengths_Bx.max().item() + max_delay_pattern
        outputs = []

        if max_len > 0:
            num_channels = self.config.data.channels
            audio_pad_value = self.config.data.audio_pad_value
            generated_codes = torch.full(
                (batch_size, max_len, num_channels),
                fill_value=audio_pad_value,
                dtype=torch.long,
                device=self.device,
            )

            for i in range(batch_size):
                start_step = dec_output.prefill_steps[i]
                actual_len = lengths_Bx[i].item() + max_delay_pattern
                if actual_len > 0:
                    tokens_to_copy = dec_output.generated_tokens[i, start_step : start_step + actual_len, :]
                    generated_codes[i, :actual_len, :] = tokens_to_copy

            if verbose:
                avg_steps = lengths_Bx.float().mean().item()
                total_duration = time.time() - total_start_time
                print(f"generate: avg steps={avg_steps:.1f}, total duration={total_duration:.3f}s")

            del dec_state

            outputs = self._generate_output(generated_codes, lengths_Bx)
        else:
            print("Warning: Nothing generated for any sequence in the batch.")
            outputs = [None] * batch_size

        return outputs if isinstance(text, list) else outputs[0]
