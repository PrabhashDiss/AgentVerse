from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, List, Optional

from . import order_registry as OrderRegistry
from .base import BaseOrder

if TYPE_CHECKING:
    from agentverse.environments import BaseEnvironment


@OrderRegistry.register("kitchen")
class KitchenOrder(BaseOrder):
    """The order for the Mother and Son Dilemma
    The agents speak in the following order:
    1. The mother speaks first.
    2. Then the son responds.
    3. This alternates until the final decision.
    """

    def get_next_agent_idx(self, environment: BaseEnvironment) -> List[int]:
        if len(environment.last_messages) == 0:
            # If the interaction just begins, we let only the mother speak
            return [0]  # Index 0 for Mother
        else:
            last_message = environment.last_messages[-1]
            sender = last_message.sender
            if sender == "Mother":
                # After the mother speaks, the son should respond
                return [1]  # Index 1 for Son
            elif sender == "Son":
                # After the son speaks, the mother should respond
                return [0]  # Index 0 for Mother
            else:
                # Default case, to handle any unexpected scenarios
                return [0]
