import torch
import numpy as np
from typing import Optional, Tuple, Union

import transformers.models.bert.modeling_bert as modeling_bert
from transformers import AutoTokenizer

from lxt.efficient import monkey_patch
from lxt.utils import clean_tokens

from . import BaseExplainer
from .explanation import Explanation
from .utils import parse_explainer_args


class LXTExplainer(BaseExplainer):
    NAME = "LXT"

    def __init__(self, model, tokenizer, model_helper: Optional[str] = None, **kwargs):
        """
        LXT Grad × Input explainer.

        Args:
            model: HuggingFace model (BertForSequenceClassification).
            tokenizer: HuggingFace tokenizer.
            model_helper: optional model helper.
        """
        super().__init__(model, tokenizer, model_helper, **kwargs)

        # Monkey-patch BERT to enable efficient relevance propagation
        monkey_patch(modeling_bert, verbose=False)

        self.model.eval()

    def compute_feature_importance(
        self,
        text: Union[str, Tuple[str, str]],
        target: Union[int, str] = 1,
        target_token: Optional[Union[int, str]] = None,
        **kwargs,
    ):
        # Sanity checks
        target_pos_idx = self.helper._check_target(target)
        target_token_pos_idx = self.helper._check_target_token(text, target_token)
        text = self.helper._check_sample(text)

        # Tokenize
        item = self._tokenize(text)
        item = {k: v.to(self.device) for k, v in item.items()}
        input_ids = item["input_ids"]

        # Get embeddings
        inputs_embeds = self.model.bert.get_input_embeddings()(input_ids).requires_grad_(True)

        # Forward pass
        logits = self.model(inputs_embeds=inputs_embeds, attention_mask=item["attention_mask"]).logits
        pred_idx = logits.argmax(dim=-1).item()

        # Backprop relevance
        grad = torch.autograd.grad(
            outputs=logits[0, target_pos_idx],
            inputs=inputs_embeds,
            retain_graph=False,
            create_graph=False,
        )[0]

        relevance = (inputs_embeds * grad).sum(dim=-1).squeeze(0).detach().cpu().numpy()

        # Normalize
        if np.max(np.abs(relevance)) > 0:
            relevance = relevance / np.max(np.abs(relevance))

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        tokens = clean_tokens(tokens)

        # Build Explanation object
        output = Explanation(
            text=text,
            tokens=tokens,
            scores=relevance,
            explainer=self.NAME,
            helper_type=self.helper.HELPER_TYPE,
            target_pos_idx=target_pos_idx,
            target_token_pos_idx=target_token_pos_idx,
            target=self.helper.model.config.id2label[target_pos_idx],
            target_token=None,  # Only relevant for token-level tasks
        )
        return output
