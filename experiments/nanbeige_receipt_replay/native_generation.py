"""Use the existing cached model without inspecting disabled capture callbacks."""
from experiments.nanbeige_reasoning.llama_cached_steer import CachedSteeringModel


class GenerationCachedModel(CachedSteeringModel):
    def _callback(self,tensor,ask,user):
        # The inherited callback returns False in this state for every tensor.
        # Keep all capture and intervention paths on the existing implementation.
        if ask and not self.enabled and self.steering is None:
            return False
        return super()._callback(tensor,ask,user)
