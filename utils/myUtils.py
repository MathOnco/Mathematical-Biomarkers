# ====================================================================================
# Various functions useful for this project
# ====================================================================================
import numpy as np
import pandas as pd
import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
from tqdm import tqdm
import string
import json
import datetime
import re
import os

# ====================================================================================
# Functions for dealing with treatment schedules
# ====================================================================================

# Helper function to load dataframe from file
def LoadPatientData(patientId, dataDir, file_extension=".txt"):
    patientDataDf = pd.read_csv(os.path.join(dataDir, "patient%.3d" % patientId + file_extension), header=None)
    # patientDataDf = pd.read_csv(dataDir, header=None)
    patientDataDf.rename(columns={0: "PatientId", 1: "Date", 2: "CPA", 3: "LEU", 4: "PSA", 5: "Testosterone",
                                  6: "CycleId", 7: "DrugConcentration"}, inplace=True)
    patientDataDf['Date'] = pd.to_datetime(patientDataDf.Date)
    patientDataDf = patientDataDf.sort_values(by="Date")
    patientDataDf['Time'] = patientDataDf[8] - patientDataDf.iloc[0, 8]
    patientDataDf['PSA_raw'] = patientDataDf.PSA
    patientDataDf['PSA'] /= patientDataDf.PSA.iloc[0]
    return patientDataDf

# Get all Patient IDs from a given directory
def GetPatientIDs(file_location):
    numbers = []
    file_names = os.listdir(file_location)
    for file_name in file_names:
        # Extract numbers using regular expression
        matches = re.findall(r'\d+', file_name)
        numbers.append(int(matches[0]))  # Assume first number is ID
    return sorted(numbers)

# Helper function to extract the treatment schedule from the data
def ConvertTDToTSFormat(timeVec,drugIntensityVec):
    treatmentScheduleList = [] # Time intervals in which we have the same amount of drug
    tStart = timeVec[0]
    currDrugIntensity = drugIntensityVec[0]
    for i,t in enumerate(timeVec):
        if drugIntensityVec[i]!=currDrugIntensity and not (np.all(np.isnan([drugIntensityVec[i],currDrugIntensity]))): # Check if amount of drug has changed
            treatmentScheduleList.append([tStart,t,currDrugIntensity])
            tStart = t
            currDrugIntensity = drugIntensityVec[i]
    treatmentScheduleList.append([tStart,timeVec[-1]+(tStart==timeVec[-1])*1,currDrugIntensity])
    return treatmentScheduleList

# Helper function to obtain treatment schedule from calibration data
def ExtractTreatmentFromDf(dataDf,treatmentColumn="DrugConcentration"):
    timeVec = dataDf['Time'].values
    nDaysPreTreatment = int(timeVec.min())
    if nDaysPreTreatment != 0: # Add the pretreatment phase if it's not already added
        timeVec = np.concatenate((np.arange(0, nDaysPreTreatment), timeVec), axis=0)
    drugIntensityVec = dataDf[treatmentColumn].values
    drugIntensityVec = np.concatenate((np.zeros((nDaysPreTreatment,)), drugIntensityVec), axis=0)
    return ConvertTDToTSFormat(timeVec, drugIntensityVec)

# Turns a treatment schedule in list format (i.e. [tStart, tEnd, DrugConcentration]) into a time series
def TreatmentListToTS(treatmentList,tVec):
    drugConcentrationVec = np.zeros_like(tVec)
    for drugInterval in treatmentList:
        drugConcentrationVec[(tVec>=drugInterval[0]) & (tVec<=drugInterval[1])] = drugInterval[2]
    return drugConcentrationVec

# Extract the date as a datetime object from a model or experiment data frame
def GetDateFromDataFrame(df):
    year, month, day, hour, minute = [df[key].values[0] for key in ['Year','Month','Day','Hour','Minute']]
    hour = 12 if np.isnan(hour) else hour
    minute = 0 if np.isnan(minute) else minute
    return datetime.datetime(int(year),int(month),int(day),int(hour),int(minute))

def TruncateDataframe(dataframe, drug_name = 'Abi'):
    """Select the first cycle from patient data.
    Done by selecting the first occurence of drug turning
    on in the 'Abi' column."""
    drug_seq = ''.join([str(int(x)) for x in dataframe[drug_name].values])
    end_ind = drug_seq.find('01')
    return dataframe.head(end_ind + 1) if end_ind >= 0 else dataframe


# ====================================================================================
# Analytics
# ====================================================================================
def calc_critical_treatment_interval(n_crit, S0, K, rS, dS, prog = 1.2, **kwargs):
    """
    Calculates maximal treatment interval that prevents premature failure for a given treatment threshold n_crit.
    Errors in the log may result from n_crit > 1.2S0
    """
    log_1 = ((K * (dS-rS) + prog*rS*S0) / (K * (dS-rS) + rS*n_crit))
    log_2 = (prog * S0) / (n_crit)
    return (np.log(log_1) / (dS-rS)) + (np.log(log_2) / (rS-dS))

def calc_critical_treatment_threshold(tau, S0, K, rS, dS, prog = 1.2, **kwargs):
    """
    Calculates maximal treatment threshold that prevents premature failure for a given treatment interval tau.
    Note this is not normalised by the initial tumour size S0.
    """
    numerator = K * (dS - rS)
    pre_exp = (numerator / (prog * S0)) + rS
    denominator = (pre_exp * np.exp(tau * (rS - dS))) - rS
    return numerator / denominator

def calc_doubling_time(kappa, dR, K, rR, R0, **kwargs):
    """
    Calculates the doubling time of a population of R0 resistant cell,
    in the presence of kappa*K sensitive cells.
    Special case of the `calc_growth_time` function.
    """
    T_log = (rR * R0) / (dR * K - (1 - kappa) * rR * K + rR * R0)
    T = (np.log(1 + T_log) - np.log(2)) / (dR - (1 - kappa) * rR)
    return T

def calc_growth_time(kappa, dR, K, rR, R0, n0, upper = None, **kwargs):
    """
    Calculates the growth time of a population of R0 resistant cell,
    in the presence of kappa*K sensitive cells.
    Defaults to computing the doubling time of the population.
    Can specify an upper limit for the population size - i.e. upper = 1.2*n0
    would compute the progression time instead.
    """
    if upper is None: upper = 2 * R0
    numerator = upper * (rR * R0 + dR * K - (1 - kappa) * rR * K)
    denominator = R0 * (upper * rR + dR * K - (1 - kappa) * rR * K)
    T = np.log(numerator / denominator) / (rR * (1 - kappa) - dR)
    return T

def calc_delta_benefit(kappa, dR, K, rR, R0, **kwargs):
    """
    Calculates the fractional change in doubling time from increasing kappa.
    """
    T_plus = calc_growth_time(kappa, dR, K, rR, R0, **kwargs)
    T_minus = calc_growth_time(0, dR, K, rR, R0, **kwargs)
    return (T_plus - T_minus) / T_minus

def calc_threshold_benefit(tau, **kwargs):
    params=kwargs
    n_crit = calc_critical_treatment_threshold(tau=tau, **params)
    return calc_delta_benefit(kappa=n_crit / params['K'], **params)

def calc_average_overshoot(tau, n_crit, **kwargs):
    # Arithmetic average
    params = kwargs
    a = params['rS'] * (1 - params['dD']) - params['dR']
    b = params['rS'] * (1 - params['dD']) / params['K']
    denom = b * n_crit - (b * n_crit - a) * np.exp(- a * tau)
    max_overshoot_pos = (a * n_crit) / denom
    return (n_crit - max_overshoot_pos) / 2  # Return average shift

def calc_atx_benefit(tau, **kwargs):
    """Calculate benefit of using an interval based optimal strategy 
    at a given time interval, relative to an unsuppressed population.
    Suppression is given by arithmetic mean of crit threshold and 
    progression limit, as the tumor oscillates between these two values.
    """
    params = kwargs
    n_crit = calc_critical_treatment_threshold(tau=tau, **params)
    n_overshoot = calc_average_overshoot(tau, n_crit, **params)
    ave_n = ((n_crit + 1.2 * params['n0'])/2 - n_overshoot)
    return calc_delta_benefit(kappa=(ave_n / params['K']), **params)

def calc_at_threshold_benefit(tau, **kwargs):
    """Calculate benefit of using optimal strategy at a given time
    interval, relative to an unsuppressed population.
    No overshooting correction is required when using a single threshold.
    """
    params = kwargs
    n_crit = calc_critical_treatment_threshold(tau=tau, **params)
    return calc_delta_benefit(kappa=(n_crit / params['K']), **params)

def predict_ct_ttp(n_lim=None, **kwargs):
    # This is in the absence of sensitive cells
    params = kwargs
    if n_lim is None: n_lim = params['n0'] * 1.2
    
    factor = params['rR'] - params['dR']
    log_num = params['rR'] - factor * (params['K'] / params['R0'])
    log_den = params['rR'] - factor * (params['K'] / n_lim)
    return np.log(log_num / log_den) / factor

def predict_atx_ttp(tau, **kwargs):
    params = kwargs
    benefit = calc_atx_benefit(tau, **params)
    ct_ttp = predict_ct_ttp(**params)
    return ct_ttp * (1 + benefit)

def predict_at_threshold_ttp(tau, **kwargs):
    params = kwargs
    benefit = calc_at_threshold_benefit(tau, **params)
    ct_ttp = predict_ct_ttp(**params)
    return ct_ttp * (1 + benefit)

def predict_drug_benefit(tau, n_crit, **kwargs):
    params = kwargs

    a = params['rS'] * (1 - params['dD']) - params['dS']
    b = params['rS'] * (1 - params['dD']) / params['K']
    zeta = calc_average_overshoot(tau, n_crit, **params)
    N_1 = n_crit - zeta; N_2 = 1.2 * params['n0'] - zeta
    T_on = np.log((N_1 * (b * N_2 - a)) / (N_2 * (b * N_1 - a))) / a
    
    a_off = params['rS'] - params['dS']; b_off = params['rS'] / params['K']
    T_off = np.log((N_2 * (b_off * N_1 - a_off)) / (N_1 * (b_off * N_2 - a_off))) / a_off
    return T_on / (T_on + T_off)
# ====================================================================================
# Misc
# ====================================================================================
def convert_ode_parameters(n0=0.75, rFrac=1e-3, cost=0, turnover=0, rS = 0.027, **kwargs):
    '''
    Converts parameters from a cost/turnover description (more interpretable, used in fitting model to 
    patient data) to the parameters used in the ODE model. Returns a dictionary that can be fed into the
    LotkaVolterraModel() class.
    
    :param n0: initial tumor density
    :param rFrac: initial resistance fraction (R0/(S0+R0))
    :param cost: resistance cost (rR = (1-cost)*rS)
    :param turnover: tumor cell death (dT = turnover*rS)
    :param rS: sensitive cell proliferation rate (used as time scale)
    '''
    return {'n0': n0, 'rS':rS, 'rR':(1-cost)*rS, 'dS':turnover*rS, 'dR':turnover*rS,
            'dD':1.5, 'K':1., 'D':0, 'theta':1, 'DMax':1.,
            'S0':n0*(1-rFrac), 'R0':n0*rFrac}

def test_progression(paramDic):
    """
    Return whether a patient with given parameters progresses in finite time under treatment
    """
    if paramDic['rR'] == 0:
        return False
    steady_state = paramDic['K'] * (1 - paramDic['dR'] / paramDic['rR'])
    return steady_state > 1.2 * paramDic['n0']

def obtain_architecture(model_name): 
    '''
    Returns dictionary of achitecture parameters from parameters file, to feed back into eval function.
    '''
    path = os.path.join("../models/", model_name, "paramDic_%s.txt"%(model_name))
    with open(path, 'r') as file:
        lines = [line.rstrip() for line in file]
        in_dict = {l.split(':')[0]:l.split(':')[1] for l in lines}  
        my_dict = {key: int(in_dict[key]) for key in ['n_inputs', 'n_values_size', 'n_values_delta', 'n_doseOptions']}
        my_dict['architecture'] = json.loads(in_dict['architecture'])
        return my_dict

def printTable(myDict, colList=None, printHeaderB=True, colSize=None, **kwargs):
    """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
    If column names (colList) aren't specified, they will show in random order.
    Author: Thierry Husson - Use it as you want but don't blame me.
    """
    if not colList: colList = list(myDict[0].keys() if myDict else [])
    myList = [colList] if printHeaderB else [] # 1st row = header
    for item in myDict: myList.append([str('%.2e'%item[col] or '') for col in colList])
    colSize = [max(map(len,col)) for col in zip(*myList)] if not colSize else colSize
    formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
    if printHeaderB: myList.insert(1, ['-' * i for i in colSize]) # Seperating line
    for item in myList: print(formatStr.format(*item))
    if kwargs.get('getColSizeB',False): return colSize
