from .accountant import IAccountant
from .gdp_accountant import GaussianAccountant
# from .rdp import RDPAccountant


__all__ = [
    "IAccountant",
    "GaussianAccountant",
    # "RDPAccountant",
]


def create_accountant(mechanism: str) -> IAccountant:
    # if mechanism == "rdp":
    #     return RDPAccountant()
    if mechanism == "gdp":
        return GaussianAccountant()

    raise ValueError(f"Unexpected accounting mechanism: {mechanism}")