import numpy as np
import pandas as pd
import os
import sys
if 'matplotlib' not in sys.modules:
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import seaborn as sns
sns.set(style="white")
sys.path.append("./")
import myUtils as utils


def plot_simulation(dataDf, plotPops=False, lineplot_kws=None,
                    plotDrug=True, plotDrugAsBar=True, drugBarPosition=0.85, drugplot_kws=None,
                    xlim=None, ylim=None, y2lim=[0,1],
                    markInitialSize=False, markProgression=False, lineYPos=None, hlines_kws={},
                    decorateAxes=True, decoratey2=True, ax=None, saveFig=False, outName="plot.png", **kwargs):
    '''
        Plot tumour trajectory under a DRL-guided treatment.
        :param dataDf: Pandas data frame with longitudinal data to be plotted.
        :param plotPops: Whether or not to plot the sensitive and resistant sub-populations.
        :param lineplot_kws: Keywords for call to sns.lineplot(). Can be used for e.g. providing a colour palette.
        :param plotDrug: Boolean; whether or not to plot the treatment schedule.
        :param plotDrugAsBar: Boolean, whether to plot drug as bar across the top (True), or as shading underneath plot.
        :param drugBarPosition: Position of the drug bar when plotted across the top.
        :param drugplot_kws: Keywords to call to plt.fill_between(). Can be used for e.g. setting the drug bar colour.
        :param xlim: x-axis limit.
        :param ylim: y-axis limit.
        :param y2lim: y2-axis limit.
        :param markInitialSize: Boolean, whether or not to draw horizontal line at initial tumour size.
        :param markProgression: Boolean, whether or not to draw horizontal line at tumour size that defines progression.
        :param lineYPos: y-position at which to plot horizontal line (if different to initial size or progression).
        :param decorateAxes: Boolean, whether or not to add labels to x- and y-axes.
        :param decorateY2: Boolean, whether or not to add labels and ticks to y2-axis.
        :param ax: matplotlib axis to plot on. If none provided creates a new figure.
        :param saveFig: Boolean, whether or not to save figure.
        :param outName: String, file name to use when saving figure (also defines output format).
        :param kwargs: Other kwargs to pass to plotting functions.
        :return:
    '''
    if ax is None: fig, ax = plt.subplots(1, 1)
    varsToPlotList = ['TumourSize']
    if plotPops: varsToPlotList += ['S', 'R']
    lineplot_kws = {} if lineplot_kws is None else lineplot_kws
    drugplot_kws = {} if drugplot_kws is None else drugplot_kws
    # currModelPredictionDf = pd.melt(dataDf, id_vars=['Time'], value_vars=varsToPlotList)
    lineplot_kws['legend'] = lineplot_kws.get('legend', False) # by default turn legend off
    lineplot_kws['lw'] = lineplot_kws.get('lw', 3)  # by default turn legend off
    lineplot_kws['palette'] = lineplot_kws.get('palette', {"TumourSize": sns.xkcd_rgb['ocean blue'], "S": "#0F4C13", "R": "#710303"})
    style_palette = {"TumourSize":"-", "S":"--", "R":"--"}
    # tmpDf = dataDf[varsToPlotList+["Time"]].melt(['Time'])
    # sns.lineplot(x="Time", y="value", hue="variable", style="variable", #dashes=style_palette,
    #                 data=tmpDf, ax=ax, **lineplot_kws)
    for var in varsToPlotList:
        sns.lineplot(x="Time", y=var, color=lineplot_kws['palette'][var],
                     data=dataDf, ax=ax, linestyle=style_palette[var], **lineplot_kws) # style="variable", ,
        if lineplot_kws.get('estimator', None) is not None:
            ax.lines[-1].set_linestyle(style_palette[var])
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # Plot the drug concentration
    if plotDrug:
        ax2 = ax.twinx()
        drugplot_kws['color'] = drugplot_kws.get('color', 'black')

        # Generate time array from longest surviving patient
        max_t_id = dataDf['ReplicateId'].loc[dataDf['Time'].idxmax()]
        time_series = dataDf[dataDf.ReplicateId==max_t_id]['Time']

        # Generate drug data as time series from list
        for id in dataDf['ReplicateId'].unique():
            df = dataDf[dataDf.ReplicateId==id]
            drugConcentrationItem = utils.TreatmentListToTS(treatmentList=utils.ExtractTreatmentFromDf(df),
                                                        tVec=time_series)
            try:
                drugConcentrationVec += drugConcentrationItem
                max_dose = max(max_dose, np.max(drugConcentrationItem))
            except UnboundLocalError:
                drugConcentrationVec = drugConcentrationItem
                max_dose = np.max(drugConcentrationItem)

        # Normalise drug concentration to 0-1 (1 = max_dose)
        numSurvivorsVec = dataDf.Time.value_counts().values
        drugConcentrationVec = np.array([drug / (max_dose * num_survivors + 1e-12)
                                         for drug, num_survivors in zip(drugConcentrationVec, numSurvivorsVec)])

        # Optional moving average of drug value
        # drugConcentrationVec = np.array(pd.Series(drugConcentrationVec).rolling(10).mean())
        # drugConcentrationVec = np.nan_to_num(drugConcentrationVec)

        if not plotDrugAsBar: # Plot as bands that run down the full size of the figure.
            drugplot_kws['alpha'] = drugplot_kws.get('alpha', 0.25)
            ax2.fill_between(x=time_series, y1=ax.get_ylim()[0], y2=drugConcentrationVec, linewidth=0.0,
                             step="post", label="Drug Concentration", **drugplot_kws)
            
        else: # Plot as bars across top of plot. Cleaner, but can be harder to see alignment with tumour trajectory.
            currDrugBarPosition = drugBarPosition
            drugBarHeight = 1-drugBarPosition
            # Rescale to make it fit within the bar at the top of the plot
            drugConcentrationVec = drugConcentrationVec * drugBarHeight + currDrugBarPosition
            drugplot_kws['alpha'] = drugplot_kws.get('alpha', 0.5)
            ax2.fill_between(time_series, currDrugBarPosition, drugConcentrationVec, linewidth=0.0,
                             step="post", label="Drug Concentration", **drugplot_kws)
            ax2.hlines(xmin=time_series.min(), xmax=time_series.max(),
                      y=currDrugBarPosition, linewidth=3, color="black")
            currDrugBarPosition += drugBarHeight
            # Line at the top of the drug bars
            ax2.hlines(xmin=time_series.min(), xmax=time_series.max(),
                          y=currDrugBarPosition, linewidth=3, color="black")
            # ax2.axis("off")
        # Format y2 axis
        if y2lim is not None: ax2.set_ylim(y2lim)
        if not decoratey2:
            ax2.set_yticklabels("")
            ax2.tick_params(right = False, top = False)

    # Draw horizontal lines (e.g. initial size)
    if markInitialSize or markProgression or (lineYPos is not None):
        xlim = ax.get_xlim()[1]
        linesToPlotList = []
        if markInitialSize: linesToPlotList.append(dataDf.loc[dataDf['Time'] == 0, 'TumourSize'])
        if markProgression: linesToPlotList.append(1.2*dataDf.loc[dataDf['Time'] == 0, 'TumourSize'])
        if lineYPos is not None: linesToPlotList.append(lineYPos)
        for yPos in linesToPlotList:
            ax.hlines(xmin=0, xmax=xlim, y=yPos, linestyles=hlines_kws.get('linestyles',":"), 
                        linewidth=hlines_kws.get('linewidth',6), color=hlines_kws.get('color','grey'))

    # Format the plot
    ax.set_xlabel("Time" if decorateAxes else "")
    ax.set_ylabel("Tumour Size" if decorateAxes else "")
    ax2.set_ylabel("Drug Concentration" if decorateAxes else "")
    ax.set_title(kwargs.get('title', ''))
    if not decoratey2:
        ax2.set_yticklabels("")
    plt.tight_layout()
    if saveFig:
        plt.savefig(outName)
        plt.close()


# ====================================================================================
# Updated Style for Plotting
# ====================================================================================

def PlotDrug(dataDf, ax, color, plotDrugAsBar=True, currDrugBarPosition = 1.2, drugBarHeight = 0.2, line_max=None, **kwargs):
    # Generate drug data as time series from list
    drugConcentrationItem = utils.TreatmentListToTS(treatmentList=utils.ExtractTreatmentFromDf(dataDf),
                                                    tVec=dataDf['Time'])
    try:
        drugConcentrationVec += drugConcentrationItem
        max_dose = max(max_dose, np.max(drugConcentrationItem))
    except UnboundLocalError:
        drugConcentrationVec = drugConcentrationItem
        max_dose = np.max(drugConcentrationItem)

    # Plot as bars across top of plot. Cleaner, but can be harder to see alignment with tumour trajectory.
    # Rescale to make it fit within the bar at the top of the plot
    drugConcentrationVec = drugConcentrationVec * drugBarHeight + currDrugBarPosition
    line_max = dataDf['Time'].max() if line_max is None else line_max
    ax.fill_between(x=dataDf['Time'], y1=currDrugBarPosition, y2=drugConcentrationVec, linewidth=0.0,
                     alpha=0.5, step="post", label="Drug Concentration", color=color, **kwargs)
    ax.hlines(xmin=dataDf['Time'].min(), xmax=line_max, y=currDrugBarPosition, linewidth=3, color="black")
    ax.hlines(xmin=dataDf['Time'].min(), xmax=line_max, y=currDrugBarPosition + drugBarHeight, linewidth=3, color="black")
        
def PlotData(dataDf, ax, feature='PSA', drugBarPosition=0.85, drugBarHeight=0.15,
             xlim=2e3, ylim=1.3, y2lim=1, markersize=10, color = 'Black', drug_col='k',
             **kwargs):
    # Plot the data
    ax.plot(dataDf.Time, dataDf[feature],
            linestyle="None", marker="+", markersize=markersize,
            color=color, markeredgewidth=2, label=feature, **kwargs)

    if drug_col:
        PlotDrug(dataDf, ax, drug_col, plotDrugAsBar = True, 
                currDrugBarPosition = drugBarPosition, drugBarHeight = drugBarHeight)

    # # Format the plot
    ax.set_xlim(0,xlim); ax.set_ylim(0, ylim)
    ax.set_xticks(np.linspace(0, xlim, 5))
    ax.tick_params(right = False, top = False)
    
def PlotSimulation(df, legend=False, drug_label="Drug Concentration", ax=None, linewidth=5, linestyles = ['-', '--', '--'],
                   drug_limit=None, colors=None, drug_col='k', plot_subpops=True, prog_line=True, **kwargs):
    if ax is None: fig, ax = plt.subplots(1, 1)
    vars = ['TumourSize', 'S', 'R'] if plot_subpops else ['TumourSize']
    for i, var in enumerate(vars):
        ax.plot(df['Time'], df[var], linewidth=linewidth, linestyle=linestyles[i], color=colors[var], **kwargs)

    PlotDrug(df, ax, drug_col, plotDrugAsBar = True, line_max=drug_limit, currDrugBarPosition = 1.25)
    
    if prog_line:
        ax.hlines(xmin=0, xmax=max(df['Time']), y=1.2, linestyles=":", linewidth=6, color='red')

    # Format the plot
    ax.set_title(kwargs.get('title', ''))
    ax.set_ylim(0, 1.4)
    ax.yaxis.get_major_ticks()[-1].set_visible(False); ax.tick_params(right = False, top = False)
    plt.tight_layout()

def convert_to_rgb_grayscale(col):
    rgb_col = to_rgb(col)
    return tuple([(0.2989 * rgb_col[0]) + (0.5870 * rgb_col[1]) + (0.1140 * rgb_col[2]) for _ in range(3)])

def redistribute_rgb(col):
    r, g, b = col
    threshold = 1 
    m = max(r, g, b)
    if m <= threshold:
        return r, g, b
    total = r + g + b
    if total >= 3 * threshold:
        return threshold, threshold, threshold
    x = (3 * threshold - total) / (3 * m - total)
    gray = threshold - x * m
    return (gray + x * r), (gray + x * g), (gray + x * b)