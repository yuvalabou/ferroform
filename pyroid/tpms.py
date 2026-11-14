"""Triply Periodic Minimal Surface (TPMS) metadata registry.

Defines supported TPMS geometries, surface-area factors, and manufacturing
characteristics for UI, validation, and infill estimation.
"""

from dataclasses import dataclass
from enum import Enum


class TPMSType(str, Enum):
    """Supported TPMS geometry types."""

    GYROID = "gyroid"
    DIAMOND = "diamond"
    PRIMITIVE = "primitive"
    I_WP = "i_wp"
    FISCHER_KOCH_S = "fischer_koch_s"
    NEOVIUS = "neovius"

    @classmethod
    def from_str(cls, val: str | None) -> "TPMSType":
        """Parse string to TPMSType with tolerant fallback."""
        if not val:
            return cls.GYROID
        cleaned = val.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == cleaned or member.name.lower() == cleaned:
                return member
        aliases = {
            "d": cls.DIAMOND,
            "schwarz_d": cls.DIAMOND,
            "p": cls.PRIMITIVE,
            "schwarz_p": cls.PRIMITIVE,
            "iwp": cls.I_WP,
            "schoen_i_wp": cls.I_WP,
            "fks": cls.FISCHER_KOCH_S,
            "fischer_koch": cls.FISCHER_KOCH_S,
            "fischerkoch": cls.FISCHER_KOCH_S,
            "neo": cls.NEOVIUS,
            "c_p": cls.NEOVIUS,
        }
        if cleaned in aliases:
            return aliases[cleaned]
        raise ValueError(f"Unknown TPMS type: {val}. Supported: {[m.value for m in cls]}")


@dataclass(slots=True, frozen=True)
class TPMSKernel:
    """Metadata and physical properties of a TPMS geometry."""

    tpms_type: TPMSType
    display_name: str
    surface_factor: float
    description: str


# Registry of TPMS kernel metadata (evaluation is implemented in pyroid_core)
_KERNELS: dict[TPMSType, TPMSKernel] = {
    TPMSType.GYROID: TPMSKernel(
        tpms_type=TPMSType.GYROID,
        display_name="Gyroid (Schoen G)",
        surface_factor=318.3,
        description="Optimal balance of self-supporting printability, isotropic modulus, and drainability.",
    ),
    TPMSType.DIAMOND: TPMSKernel(
        tpms_type=TPMSType.DIAMOND,
        display_name="Diamond (Schwarz D)",
        surface_factor=384.0,
        description="Highest specific stiffness and energy absorption; stretching-dominated sheet deformation.",
    ),
    TPMSType.PRIMITIVE: TPMSKernel(
        tpms_type=TPMSType.PRIMITIVE,
        display_name="Primitive (Schwarz P)",
        surface_factor=235.0,
        description="Simple cubic topology with open axial channels; high axial stiffness, easy flow through axes.",
    ),
    TPMSType.I_WP: TPMSKernel(
        tpms_type=TPMSType.I_WP,
        display_name="I-WP (Schoen I-WP)",
        surface_factor=358.0,
        description="BCC interlocking package with very high crushing plateau and multi-directional strength.",
    ),
    TPMSType.FISCHER_KOCH_S: TPMSKernel(
        tpms_type=TPMSType.FISCHER_KOCH_S,
        display_name="Fischer-Koch S",
        surface_factor=396.0,
        description="Chiral labyrinth with twisted channels; high tortuosity and compact surface area.",
    ),
    TPMSType.NEOVIUS: TPMSKernel(
        tpms_type=TPMSType.NEOVIUS,
        display_name="Neovius (C(P))",
        surface_factor=512.0,
        description="14-face polyhedra with ultra-high surface area; best for catalytic carriers and micro-exchangers.",
    ),
}


def get_tpms_kernel(tpms: str | TPMSType | None) -> TPMSKernel:
    """Retrieve TPMSKernel instance by type or string name."""
    if isinstance(tpms, TPMSType):
        return _KERNELS[tpms]
    return _KERNELS[TPMSType.from_str(tpms)]


def get_all_tpms_kernels() -> list[TPMSKernel]:
    """Return all registered TPMS kernels."""
    return list(_KERNELS.values())
