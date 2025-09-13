import torch 
from .conditional_flow_matching import ConditionalTopologicalFlowMatcher


def time_dependent_loss(ut, ut_pred, t, fm: ConditionalTopologicalFlowMatcher) -> torch.Tensor:
    kappa = fm._bridge_kappa(t)
    vt, vt_pred = fm.ft.transform(ut), fm.ft.transform(ut_pred)
    return torch.mean((kappa * (vt - vt_pred)) ** 2)


def time_independent_loss(ut, ut_pred, t, fm: ConditionalTopologicalFlowMatcher) -> torch.Tensor:
    return torch.mean((ut - ut_pred) ** 2)


class TimeDependentTopologicalLoss(torch.nn.Module):
    def __init__(self, fm: ConditionalTopologicalFlowMatcher):
        super().__init__()
        self.fm = fm
    
    def forward(self, ut, ut_pred, t) -> torch.Tensor:
        return time_dependent_loss(ut, ut_pred, t, self.fm)


class TimeIndependentTopologicalLoss(torch.nn.Module):
    def __init__(self, fm: ConditionalTopologicalFlowMatcher):
        super().__init__()
        self.fm = fm
    
    def forward(self, ut, ut_pred, t) -> torch.Tensor:
        return time_independent_loss(ut, ut_pred, t, self.fm)


def build_loss(fm: ConditionalTopologicalFlowMatcher, name: str = 'time_dependent') -> callable:
    if name == 'time_dependent':
        return TimeDependentTopologicalLoss(fm)
    elif name == 'time_independent':
        return TimeIndependentTopologicalLoss(fm)
    else:
        raise ValueError(f"Invalid name: {name}")