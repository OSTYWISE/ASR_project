import torch
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def forward(
        self, log_probs, log_probs_length, text_encoded, text_encoded_length, **batch
    ) -> dict[str, torch.Tensor]:
        log_probs_t = torch.transpose(log_probs, 0, 1)  # T x B x V (T - time, sequence_length, B - batch_size, V - vocab_size)
        loss = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length.to(torch.int32),
            target_lengths=text_encoded_length,
        )

        return {"loss": loss}