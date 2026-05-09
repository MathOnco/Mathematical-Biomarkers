# Generates parameter values from clinical trial data
# Code is property of Maximillian Strobl (2020), can be accessed at
# https://github.com/MathOnco/AT_costOfResistance_LVModel/blob/master/fitPatients.py


# ------------------------ Imports ------------------------
import numpy as np
import sys
import os
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from itertools import product
import multiprocessing as mp
import re
from lmfit import minimize, Parameters

sys.path.append('../utils')
import LotkaVolterraModel as lvm
import myUtils as utils
from fittingUtils import residual, PerturbParams, ComputeRSquared, LoadPatientData, PatientToOutcomeMap, \
    GetBestFit, GenerateFitSummaryDf, GenerateFitSummaryDf_AllPatients, PlotParameterDistribution_PatientCohort, \
    PlotFits, PlotParameterDistribution_SinglePatient, QQPlot
# from Bruchovsky_et_al.bruchovsky_patients import LoadPatientData  # for .txt files

# Format plots
sns.set(style="white",
        font_scale=1.5,
        rc={'figure.figsize':(12,6)})
# ================================ Script setup ==========================================
dataDir = "patients/Bruchovsky_et_al/Prog_truncated_data_n1"
patientIdList = [int(re.findall(r'\d+',x)[0]) for x in os.listdir(dataDir)]
modelList = ["3params_Capacity_critCost_aveR0", "3params_Capacity_aveCost_aveR0", "3params_Capacity_noCost_aveR0",
             "2params_critCost_aveR0", "2params_aveCost_aveR0", "2params_noCost_aveR0"]
modelList = ["2params_aveCost_aveR0_critTurnover", "2params_noCost_aveR0_critTurnover"]
nFits = 5
perturbICs = False # was True
runParallel = False
nProcesses = 12
outDir = 'patients/Bruchovsky_et_al/Prog_truncated_fits_n1'

# Parameterise fitting algorithm
eps_data = 1.
solver_kws={'method':'LSODA', 'absErr':1.0e-8, 'relErr':1.0e-6, 'suppressOutputB':True, 't_end': 5000}
optimiser_kws = {'method':'least_squares', 'xtol':1e-8, 'ftol':1e-8,
                 'nan_policy':'omit', 'verbose':0}
params = Parameters()
params.add('rS', value=0.027, min=1e-4, max=0.1, vary=False)
params.add('cost', min=0, max=1., vary=True)
params.add('rR', expr='(1-cost)*rS', vary=False)
params.add('turnover', min=0, max=1., vary=True)
params.add('dS', expr='turnover*rS', vary=False) # Constrain d<r
params.add('dR', expr='dS', vary=False) # Constrain dS = dR
params.add('dD', value=1.5, min=1, max=2, vary=False)
params.add('K', value=1, vary=False)
params.add('D', value=1, vary=False)
params.add('theta', value=1, vary=False)
params.add('DMax', value=1, vary=False)
params.add('n0', min=0.1, max=1., vary=True)
params.add('fR', min=1e-5, max=0.25, vary=True)
params.add('S0', expr='n0*(1-fR)', vary=False)
params.add('R0', expr='n0*fR', vary=False)

# ============================= Auxillary Functions ======================================

def FitModel(job):
    patientId, fitId, params, outDir = job['patientId'], job['fitId'], job['params'], job['outDir']
    dataDf = LoadPatientData(patientId,dataDir)
    currOutDir = os.path.join(outDir,"patient%d"%(patientId))
    if os.path.isfile(os.path.join(currOutDir,"fitObj_patient_%d_fit_%d.p"%(patientId, fitId))): return 0
    os.makedirs(currOutDir, exist_ok=True)
    seed = int.from_bytes(os.urandom(4), byteorder='little')
    np.random.seed(seed)
    tmpModel = lvm.LotkaVolterraModel()
    if perturbICs: params = PerturbParams(params)
    try:
        fitObj = minimize(residual, params, args=(0, dataDf, eps_data, tmpModel, "PSA", solver_kws),**optimiser_kws)
        # Plot best fit
        myModel = lvm.LotkaVolterraModel()
        myModel.SetParams(**fitObj.params.valuesdict())
        myModel.paramDic['K'] = fitObj.params.valuesdict()['K']
        myModel.Simulate(treatmentScheduleList=utils.ExtractTreatmentFromDf(dataDf),max_step=1,**solver_kws)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plt.plot(dataDf.Time, dataDf.PSA, linestyle='none', marker='x')
        myModel.Plot(plotPops=True, ymin=0, legendB=False, ax=ax)
        plt.savefig(os.path.join(currOutDir,"patient_%d_fit_%d.png" % (patientId, fitId)))
        plt.close()

        # Save fit
        fitObj.patientId = patientId
        fitObj.fitId = fitId
        fitObj.seed = seed
        fitObj.eps_data = eps_data
        fitObj.rSq = ComputeRSquared(fitObj,dataDf)
        pickle.dump(obj=fitObj, file=open(os.path.join(currOutDir,"fitObj_patient_%d_fit_%d.p"%(patientId, fitId)), "wb"))
    except Exception as e:
        print(dataDf.columns.values)
        print(e)
        pass

def PlotModel(file):
    fileo = open(file,'rb')
    fitObj = pickle.load(fileo)
    fileo.close()
    currOutDir = os.path.dirname(file)
    plot_name = os.path.splitext(os.path.basename(file))[0].lstrip("fitObj_")

    patientId = int(re.findall('\d+|$', plot_name)[0])
    dataDf = LoadPatientData(patientId,dataDir)

    # Plot best fit
    myModel = lvm.LotkaVolterraModel()
    myModel.SetParams(**fitObj.params.valuesdict())
    myModel.paramDic['K'] = fitObj.params.valuesdict()['K']
    myModel.Simulate(treatmentScheduleList=utils.ExtractTreatmentFromDf(dataDf),max_step=1,**solver_kws)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plt.plot(dataDf.Time, dataDf.PSA, linestyle='none', marker='x', label='Patient Data')
    myModel.Plot(plotPops=True, ymin=0, legendB=False, ax=ax, legend=False, drug_label="Treatment Periods",
                 palette= {"TumourSize": sns.xkcd_rgb['ocean blue'], "S": "#229429", "R": "#b50e0e"})
    plt.savefig(os.path.join(currOutDir, plot_name) + "_nk.png")
    plt.close()

    # qq_name = os.path.join(currOutDir, ("patient" + str(patientId))) + "_qq.png"
    # QQPlot(dataDf, fitObj, decorate=True, outName=qq_name)

    PlotParameterDistribution_SinglePatient(patientId, os.path.split(currOutDir)[0])
    plt.savefig(os.path.join(currOutDir, ("patient" + str(patientId))) + "_params.png")


# ================================= Main ================================================
pool = mp.Pool(processes=nProcesses,maxtasksperchild=1) if runParallel else None

print("started")
jobList = []
for model in modelList:
    currParams = params.copy()
    for setting in model.split("_"):
        if setting == "noCost":
            currParams['cost'].set(value=0,vary=False)
        if setting == "aveCost":
            currParams['cost'].set(value=0.57,vary=False)
        if setting == "critCost":
            currParams['cost'].set(min=0, max=0, expr='1-(turnover/(1-(1.2*n0)/k))',vary=False)
        if setting == "noTurnover":
            currParams['turnover'].set(value=0,vary=False)
        if setting == "Capacity":
            currParams['K'].set(min=0, max=10, vary=True)
        if setting == "aveR0":
            currParams['R0'].set(value=0,vary=False)
        if setting == "critTurnover":
            currParams.add('delta_n', min=0, vary=True)
            currParams['turnover'].set(min=0, max=1, expr='(1-cost)*(1-(1.2*(n0+delta_n))/k)', vary=False)
    currOutDir = os.path.join(outDir,model)
    for patientId,fitId in tqdm(product(patientIdList,range(nFits)),disable=runParallel):
        job = {'patientId':patientId, 'fitId':fitId, 'params':currParams, 'outDir':currOutDir}
        jobList.append(job)
        if not runParallel: FitModel(job)
if runParallel: list(tqdm(pool.imap(FitModel, jobList), total=len(jobList)))

# Analyse data
paramList = ["cost","turnover","n0","fR"]
for model in modelList:
    print(model)
    fitDir = os.path.join(outDir,model)
    dataToAnalyse = GenerateFitSummaryDf_AllPatients(patientIdList,fitDir,dataDir=dataDir)
    dataToAnalyse.to_csv(os.path.join(fitDir, "fitPatientsDf_all.csv"))

    # # Plot all fits
    # fig, axList = plt.subplots(6, 3, sharex=True, sharey=True, figsize=(18, 20))
    # for i, patientId in enumerate(dataToAnalyse.PatientId.unique()):
    #     fit = GetBestFit(patientId, fitDir=fitDir)
    #     PlotFits(fits=[fit.fitId], dataDf=LoadPatientData(patientId,dataDir=dataDir), fitDir=fitDir,
    #              solver_kws={**solver_kws,'max_step':1.},
    #              ylim=1.5, titleStr="P%d; R² = %1.2f" % (patientId, fit.rSq), ax=axList.flatten()[i])

    # fig.delaxes(axList[-1][-1])  # Use to remove blank axes
    # ax = fig.add_subplot(111, frameon=False)
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # fig.legend(axList[0][0].lines, ['Total Tumour', 'Resistant', 'Sensitive', 'Patient Data'],
    #            loc='lower right', ncol=len(axList[0][0].lines), prop={'size':20})
    # plt.xlabel("Time in Days", fontdict={'fontsize':28})
    # plt.ylabel("PSA (Normalised)", fontdict={'fontsize':28})
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1, hspace=0.3)
    # ax.xaxis.set_label_coords(.5, -0.03)
    # ax.yaxis.set_label_coords(-.075, .5)
    # plt.savefig(os.path.join(fitDir,"overview.pdf"))

    # Plot the parameters
    PlotParameterDistribution_PatientCohort(dataToAnalyse,paramList=paramList,fitDir=fitDir,
                                            outName=os.path.join(fitDir,"paramDist.pdf"))


# PlotModel("patients/eLife_Trial/at_patients/fits/5params_Capacity/patient1014/fitObj_patient_1014_fit_0.p")
# GenerateFitSummaryDf(12, "patients/Bruchovsky_et_al/Complete_fits/4params").to_csv("test.csv")