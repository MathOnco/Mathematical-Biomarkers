# ====================================================================================
# Classes to simulate 2-population tumour models
# ====================================================================================
import sys
import numpy as np
from math import exp
sys.path.append("./")
from odeModelClass import ODEModel

# ======================== House Keeping Funs ==========================================
def create_model(modelName,**kwargs):
    funList = {"LotkaVolterraModel":LotkaVolterraModel,
               "CompetitionModel":CompetitionModel}
    return funList[modelName](**kwargs)

# ======================= Models =======================================
class LotkaVolterraModel(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "LotkaVolterraModel"
        self.paramDic = {**self.paramDic,
                        'rS': 0.027,
                        'rR': 0.027,
                        'K': 1,
                        'dD': 1.5,
                        'dS': 0.,
                        'dR': 0.,
                        'S0': 0.74,
                        'R0':0.01,
                        'DMax':1}
        self.stateVars = ['S', 'R']

    # The governing equations
    def ModelEqns(self, t, uVec):
        S, R, D = uVec
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = self.paramDic['rS'] * (1 - (S+R)/self.paramDic['K']) * (1-self.paramDic['dD']*D) * S - self.paramDic['dS']*S
        dudtVec[1] = self.paramDic['rR'] * (1 - (S+R)/self.paramDic['K']) * R - self.paramDic['dR']*R
        dudtVec[2] = 0
        return (dudtVec)
    
class CompetitionModel(ODEModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "CompetitionModel"
        self.paramDic = {**self.paramDic,
                        'rS': 0.027,
                        'rR': 0.027,
                        'K': 1,
                        'dD': 1.5,
                        'dS': 0.,
                        'dR': 0.,
                        'S0': 0.74,
                        'R0':0.01,
                        'DMax':1}
        self.stateVars = ['S', 'R']

    # The governing equations
    def ModelEqns(self, t, uVec):
        S, R, D = uVec
        dudtVec = np.zeros_like(uVec)
        dudtVec[0] = 0
        dudtVec[1] = self.paramDic['rR'] * (1 - (S+R)/self.paramDic['K']) * R - self.paramDic['dR']*R
        dudtVec[2] = 0
        return (dudtVec)
    