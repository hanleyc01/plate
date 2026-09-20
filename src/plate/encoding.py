"""Encoding parsed representations as VSAs."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from . import memory, syntax, vsa


@dataclass
class Encoder[V: vsa.VSA[Any]]:
    """Encoder class for transforming `syntax.Program`'s into `VSA`."""

    vsa_type: type[V]
    dim: int
    cleanup_memory: memory.VSACleanup[V]
    associative_memory: memory.VSA_Associative[V]
    integer_encoding_scheme: Literal["list", "residue"] = field(
        default="residue"
    )
    list_encoding_scheme: Literal["rfp", "kanerva"] = field(default="kanerva")
