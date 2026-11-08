"""Full-model interface reserved for assembly after the Mamba exercises."""

from torch import Tensor, nn


class NemotronForCausalLM(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        raise NotImplementedError("Assemble hybrid blocks after Mamba review")

    def forward(self, input_ids: Tensor) -> Tensor:
        """Token IDs [batch, length] -> logits [batch, length, vocabulary]."""
        raise NotImplementedError("Model assembly milestone")
