'''
Module to handle mixing two interaction matrices
'''

from torch import Tensor


class InteractionMatrixMixer():
    '''class that combines the interaction matrices at the current and desired poses.
    '''

    def requires_Lid(self) -> bool:
        '''whether the mixer requires the interaction matrix at the desired pose'''
        return False

    def requires_Li(self) -> bool:
        '''whether the mixer requires the interaction matrix at the current pose'''
        return False
    
    def compute_final_L(self, Li: Tensor, Lid: Tensor) -> Tensor:
        '''Combines both interaction matrices to form the final interaction matrix'''
        pass
    def reset(self) -> None:
        '''For stateful mixers, resets the state'''
        pass

class DesiredInteractionMatrix(InteractionMatrixMixer):
    '''Class that does not "mix", but simply uses the interaction matrix at the desired pose'''
    def requires_Lid(self):
        return True
    def compute_final_L(self, Li, Lid):
        return Lid
class CurrentInteractionMatrix(InteractionMatrixMixer):
    '''Class that does not "mix", but simply uses the interaction matrix at the current pose
    This is the classical frame of VS
    '''

    def requires_Li(self):
        return True
    def compute_final_L(self, Li, Lid):
        return Li
class AverageCurrentAndDesiredInteractionMatrices(InteractionMatrixMixer):
    '''Mixer that averages both interaction matrices, as is done in Efficient Second Order Minimisation'''
    def requires_Li(self):
        return True
    def requires_Lid(self):
        return True
    def compute_final_L(self, Li, Lid):
        return (Li + Lid) / 2.0

class EMACurrentInteractionMatrix(InteractionMatrixMixer):
    '''
    Mixer that computes an exponential moving average of the current interaction matrix
    '''
    def __init__(self, momentum):
        self.momentum = momentum
        self.running_Li = None
    def requires_Li(self):
        return True
    def compute_final_L(self, Li, Lid):
        if self.running_Li is None:
            self.running_Li = Li
        else:
            self.running_Li = (1 - self.momentum) * self.running_Li + self.momentum * Li
        return self.running_Li
    def reset(self):
        self.running_Li = None