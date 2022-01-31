# import python libraries
import pandas as pd
import glob
import os
import copy
import numpy as np
import path
import psytrack as psy
import sys
import matplotlib.pyplot as plt

# !pip install ibllib==1.4.7
from oneibl.onelight import ONE

def listFiles(fpath):
    pathoutput = fpath + os.sep + 'fileList.csv'
    allFiles = set(glob.glob(fpath+'/**/*.xlsx', recursive=True)) # get all the excel file in folder
    settingFiles = set(glob.glob(fpath+'/**/settings.xlsx', recursive = True)) # get all the settings file

    wdilFiles = allFiles - settingFiles # exclude the settings files
    wdilFiles = list(wdilFiles) # convert the set to a list

    print('alllFiles: ', len(allFiles), ' wdilFiles: ', len(wdilFiles))
    print(wdilFiles)

    # important to sort the list to obtain proper sequence 
    wdilFiles.sort()
    wdilFiles = pd.DataFrame({'file': wdilFiles})
    wdilFiles.to_csv(pathoutput)


    return print(wdilFiles)
    return print('saved: ', pathoutput)

def formatWDILfile(path):
    '''
    Function to concatenate all the wdil file into one 
    to be able to run psytrack

    Args:
    path (str): with all the files 
    '''

    allFiles = set(glob.glob(path+'/**/*.xlsx', recursive=True)) # get all the excel file in folder
    settingFiles = set(glob.glob(path+'/**/settings.xlsx', recursive = True)) # get all the settings file

    wdilFiles = allFiles - settingFiles # exclude the settings files
    wdilFiles = list(wdilFiles) # convert the set to a list

    print('alllFiles: ', len(allFiles), ' wdilFiles: ', len(wdilFiles))
    print(wdilFiles)

    # important to sort the list to obtain proper sequence 
    wdilFiles.sort()

    allDat = [] # create an empty object to 
    sID = []
    
    for i in wdilFiles:
        print(i)
        tmp = pd.read_excel(i) # read excel file

        ## section to check with previous id and assign absolute order number
        tmpsID = i.split(os.sep)[-3]
        if tmpsID == sID:
            # print('same')
            k += 1
        else:
            k=0
        # print(k)


        sID = i.split(os.sep)[-3] # add a column with the id
        sessionDate = i.split(os.sep)[-2]
        tmp['sID'] = sID
        tmp['sessionDate'] = sessionDate
        tmp['session'] = k
        # absSession = # obtain the absolute session number

        allDat.append(tmp)
    allDat = pd.concat(allDat)

    return allDat

def codingDatFile(allDat):
    '''
    Function to code all the wdil file into one 
    to be able to run psytrack

    Args:
    allDat
    '''

    allDat['choice'] = allDat['Lick?']
    allDat = allDat.rename(columns={'Trial#':'trial'})
    ## establish what are the hit 
    ## in this case the establishment of hit correspond to true hit:
    ##      lick and was a Go # if Go=1 and Correct =1 --> CorrectCat ==2 and is a hit
    ## as well as correct rejection:
    ##      with held and was a no Go # if Go=0 and Correct =0 --> CorrectCat ==0 and is a hit
    allDat['CorrectCat']  = allDat['choice'] + allDat['Correct?'] 
    allDat['hit'] = np.where((allDat['CorrectCat']==2) | (allDat['CorrectCat']==0),1,0)

    ## those could be useful for modeling see
    ## comments on Figure F3b use of bias
    allDat['Go'] = allDat['Go/NoGo']
    allDat['NoGo'] = abs(allDat['Go/NoGo']-1)

    return allDat

def getRat(subject, first=20000, cutoff=50):

    df = RAT_DF[RAT_DF['subject_id']==subject]  # restrict dataset to single subject
    df = df[:first]  # restrict to "first" trials of data
    # remove sessions with fewer than "cutoff" valid trials
    df = df.groupby('session').filter(lambda x: len(x) >= cutoff)   

    # Normalize the stimuli to standard normal
    s_a = (df["s_a"] - np.mean(df["s_a"]))/np.std(df["s_a"])
    s_b = (df["s_b"] - np.mean(df["s_b"]))/np.std(df["s_b"])
    
    # Determine which trials do not have a valid previous trial (mistrial or session boundary)
    t = np.array(df["trial"])
    prior = ((t[1:] - t[:-1]) == 1).astype(int)
    prior = np.hstack(([0], prior))

    # Calculate previous average tone value
    s_avg = (df["s_a"][:-1] + df["s_b"][:-1])/2
    s_avg = (s_avg - np.mean(s_avg))/np.std(s_avg)
    s_avg = np.hstack(([0], s_avg))
    s_avg = s_avg * prior  # for trials without a valid previous trial, set to 0

    # Calculate previous correct answer
    h = (df["correct_side"][:-1] * 2 - 1).astype(int)   # map from (0,1) to (-1,1)
    h = np.hstack(([0], h))
    h = h * prior  # for trials without a valid previous trial, set to 0
    
    # Calculate previous choice
    c = (df["choice"][:-1] * 2 - 1).astype(int)   # map from (0,1) to (-1,1)
    c = np.hstack(([0], c))
    c = c * prior  # for trials without a valid previous trial, set to 0
    
    inputs = dict(s_a = np.array(s_a)[:, None],
                  s_b = np.array(s_b)[:, None],
                  s_avg = np.array(s_avg)[:, None],
                  h = np.array(h)[:, None],
                  c = np.array(c)[:, None])

    dat = dict(
        subject = subject,
        inputs = inputs,
        s_a = np.array(df['s_a']),
        s_b = np.array(df['s_b']),
        correct = np.array(df['hit']),
        answer = np.array(df['correct_side']),
        y = np.array(df['choice']),
        dayLength=np.array(df.groupby(['session']).size()),
    )
    return dat

def getMouse(subject, p=5):
    df = MOUSE_DF[MOUSE_DF['subject']==subject]   # Restrict data to the subject specified
    
    cL = np.tanh(p*df['contrastLeft'])/np.tanh(p)   # tanh transformation of left contrasts
    cR = np.tanh(p*df['contrastRight'])/np.tanh(p)  # tanh transformation of right contrasts
    inputs = dict(cL = np.array(cL)[:, None], cR = np.array(cR)[:, None])

    dat = dict(
        subject=subject,
        lab=np.unique(df["lab"])[0],
        contrastLeft=np.array(df['contrastLeft']),
        contrastRight=np.array(df['contrastRight']),
        date=np.array(df['date']),
        dayLength=np.array(df.groupby(['date','session']).size()),
        correct=np.array(df['feedbackType']),
        answer=np.array(df['answer']),
        probL=np.array(df['probabilityLeft']),
        inputs = inputs,
        y = np.array(df['choice'])
    )
    
    return dat

def convertToDictRat(allDat, subject, first=20000, cutoff=50):

    '''
    equivalent to the function getRat from the paper see above and here https://tinyurl.com/PsyTrack-colab
    '''

    df = allDat[allDat['sID']==subject]  # restrict dataset to single subject
    df = df[:first] # restrict to "first" trials of data
    # # remove sessions with fewer than "cutoff" valid trials
    # df = df.groupby('session').filter(lambda x: len(x) >= cutoff)   

    # Determine which trials do not have a valid previous trial (mistrial or session boundary)
    t = np.array(df["trial"])
    prior = ((t[1:] - t[:-1]) == 1).astype(int)
    prior = np.hstack(([0], prior))

    # Calculate previous correct answer
    h = (df["Correct?"][:-1] * 2 - 1).astype(int)   # map from (0,1) to (-1,1)
    h = np.hstack(([0], h))
    h = h * prior  # for trials without a valid previous trial, set to 0
    
    # Calculate previous choice
    c = (df["choice"][:-1] * 2 - 1).astype(int)   # map from (0,1) to (-1,1)
    c = np.hstack(([0], c))
    c = c * prior  # for trials without a valid previous trial, set to 0
    
    ## note here that it could be useful to have different stimulus values 
    ## important to respect the dictionary psy.COLORS hence the name of the specific names of the inputs
    inputs = dict(s1 = np.array(df['Go/NoGo'])[:, None], 
                  h = np.array(h)[:, None],
                  c = np.array(c)[:, None])

    dat = dict(
        subject = subject,
        inputs = inputs,
        s1 = np.array(df['Go/NoGo']), # correspond to the go/noGo stim
        correct = np.array(df['hit']), # hit correspond to hit and correct rejection
        answer = np.array(df['Correct?']), #this is the answer 
        y = np.array(df['choice']), #this correspond to the Lick
        dayLength=np.array(df.groupby(['session']).size()),
    )

    return dat

def convertToDictMouse(allDat, subject, first=20000, cutoff=50):

    '''
    equivalent to the function getRat from the paper see above and here https://tinyurl.com/PsyTrack-colab
    '''

    df = allDat[allDat['sID']==subject]  # restrict dataset to single subject
    df = df[:first] # restrict to "first" trials of data
    # # remove sessions with fewer than "cutoff" valid trials
    # df = df.groupby('session').filter(lambda x: len(x) >= cutoff)   

    # Determine which trials do not have a valid previous trial (mistrial or session boundary)
    t = np.array(df["trial"])
    prior = ((t[1:] - t[:-1]) == 1).astype(int)
    prior = np.hstack(([0], prior))

    # Calculate previous correct answer
    h = (df["Correct?"][:-1] * 2 - 1).astype(int)   # map from (0,1) to (-1,1)
    h = np.hstack(([0], h))
    h = h * prior  # for trials without a valid previous trial, set to 0
    
    # Calculate previous choice
    c = (df["choice"][:-1] * 2 - 1).astype(int)   # map from (0,1) to (-1,1)
    c = np.hstack(([0], c))
    c = c * prior  # for trials without a valid previous trial, set to 0
    
    # note here that it could be useful to have different stimulus values 
    inputs = dict(stim = np.array(df['Go/NoGo'])[:, None], 
                  h = np.array(h)[:, None],
                  c = np.array(c)[:, None])

    dat = dict(
        subject = subject,
        inputs = inputs,
        stim = np.array(df['Go/NoGo']), # correspond to the go/noGo stim
        correct = np.array(df['hit']), # hit correspond to hit and correct rejection
        answer = np.array(df['Correct?']), #this is the answer 
        y = np.array(df['choice']), #this correspond to the Lick
        dayLength=np.array(df.groupby(['session']).size()),
    )

    return dat

def tpath(mypath, shareDrive = 'Y'):
    '''
    path conversion to switch form linux to windows platform with define drive
    Args:
    mypath (str): path of the file of interest
    shareDrive (str): windows letter of the shared folder
    '''
    if ('google.colab' in str(get_ipython())) or sys.platform == 'win32':
         myRoot = shareDrive+':'      
    else:
        myRoot = '/run/user/1000/gvfs/smb-share:server=ishtar,share=millerrumbaughlab'


    newpath = myRoot+os.sep+mypath

    return newpath

def psyCompute(allDat, sID):
    fname = SPATH+os.sep+str(sID)+'_fig5b_data.npz'

    ## either load or generate the data
    if glob.glob(fname) == [fname]:
        dat = np.load(fname, allow_pickle=True)['dat'].item()
    else:
        ## convert the data
        outData = convertToDictRat(allDat, sID)
        new_dat = psy.trim(outData, START=0, END=12500)

        # here weights could be adjusted 
        weights = {'bias': 1, 's1': 1, 'h': 1, 'c': 1}
        K = np.sum([weights[i] for i in weights.keys()])
        # hyper guess are kept with default value as in the paper
        hyper_guess = {
         'sigma'   : [2**-5]*K,
         'sigInit' : 2**5,
         'sigDay'  : [2**-4]*K,
          }
        optList = ['sigma', 'sigDay']

        hyp, evd, wMode, hess_info = psy.hyperOpt(new_dat, hyper_guess, weights, optList)

        dat = {'hyp' : hyp, 'evd' : evd, 'wMode' : wMode, 'W_std' : hess_info['W_std'],
               'weights' : weights, 'new_dat' : new_dat}

        # Save interim result
        np.savez_compressed(SPATH+os.sep+str(sID)+'_fig5b_data.npz', dat=dat)

        # save the figure
        fig = psy.plot_weights(dat['wMode'], dat['weights'], days=dat['new_dat']["dayLength"], 
                               errorbar=dat['W_std'], figsize=(4.75,1.4))
        # plt.xlabel(None); plt.ylabel(None)
        # plt.subplots_adjust(0,0,1,1) 
        plt.savefig(SPATH +os.sep+str(sID)+ "Fig5b.pdf")

def plot_all(all_labels, all_w, Weights, figsize):
    fig = plt.figure(figsize=figsize)
    Weights = [Weights] if type(Weights) is str else Weights
    avg_len=6000 # this needs to be truncated for the average for the array to have same dimensions
    for i, W in enumerate(Weights):
        print(i,W)
        avg = []
        for i in np.arange(0,len(all_w),1):
            print(i)
            bias_ind = np.where(all_labels[i] == W)[0][-1]
            bias_w = all_w[i][bias_ind]
            avg += [list(bias_w[:avg_len]) + [np.nan]*(avg_len - len(bias_w[:avg_len]))]
            colors = psy.COLORS
            plt.plot(bias_w, color=colors[W], alpha=0.2, lw=1, zorder=2+i)
        plt.plot(np.nanmean(avg, axis=0), color=colors[W], alpha=0.8, lw=2.5, zorder=5+i)

    plt.axhline(0, color="black", linestyle="--", lw=1, alpha=0.5, zorder=1)
    plt.tight_layout()
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.xlim(0, 6000)
    # plt.ylim(-2.5, 2.5)
    return fig

def plotLabelsandW(geno, SPATH):
    '''
    this function will output all the labels and wheight for a givien genotype
    based on the corresponding files and select the genotype of interest

    args:
    corresp(pd.DataFrame): data frame with file path sID and geno
    geno(str): geno of intrest either 'wt' or 'het'
    '''
    
    ## have the corresponding file generatede 
    npzFiles = glob.glob(SPATH+os.sep+'*.npz')
    corresp = pd.DataFrame({'fname':npzFiles})
    corresp['sID'] = corresp['fname'].str.split(os.sep).str[-1].str.split('_').str[0].astype(int)
   
    ## check and implement the genotypes
    try: geno
    except: geno = None
    if geno is None:
        geno = pd.read_csv(os.sep.join(SPATH.split(os.sep)[:-2])+os.sep+'animals.csv')

    corresp = pd.merge(corresp, geno, on='sID')

    ## 
    cWTorHet = corresp[corresp['geno']=='wt']
    
    all_labels = []
    all_w = []

    for i,j in cWTorHet.iterrows():
        print(i,j)
        rat = np.load(j['fname'], allow_pickle=True)['dat'].item()
        
        labels = []
        for j in sorted(rat['weights'].keys()):
            labels += [j]*rat['weights'][j]
            
        all_labels += [np.array(labels)]
        all_w += [rat['wMode']] 


    myFigsize = (3.6,1.8)
    plot_all(all_labels, all_w, ["s1"], myFigsize)
    plt.ylim(-1, 15)
    # plt.subplots_adjust(0,0,1,1) 
    # plt.gca().set_yticks([-2,0,2])
    # plt.gca().set_xticklabels([])
    plt.savefig(SPATH +os.sep+ geno+ "Fig6a.pdf")

    plot_all(all_labels, all_w, ["bias"], myFigsize)
    plt.ylim(-7, 2)
    # plt.gca().set_yticks([-2,0,2])
    # plt.gca().set_xticklabels([])
    # plt.gca().set_yticklabels([])
    # plt.subplots_adjust(0,0,1,1) 
    plt.savefig(SPATH +os.sep+ geno+ "Fig6b.pdf")

    plot_all(all_labels, all_w, ["h"], myFigsize)
    plt.ylim(-1, 1)
    # # plt.gca().set_yticklabels([])
    # plt.subplots_adjust(0,0,1,1) 
    plt.savefig(SPATH +os.sep+ geno+ "Fig6d.pdf")


    plot_all(all_labels, all_w, ["c"], myFigsize)
    plt.ylim(-1, 1)
    # plt.gca().set_yticklabels([])
    # plt.gca().set_xticklabels([])
    # plt.subplots_adjust(0,0,1,1) 
    plt.savefig(SPATH +os.sep+ geno+ "Fig6e.pdf")

###### inputs

## from the IBL mouse data - figure 3
ibl_data_path = tpath(r'Talks\2022-01-13 - pillow\Figures\ibl-behavioral-data-Dec2019')
ibl_mouse_data_path = pd.read_csv(ibl_data_path+os.sep+"ibl_processed.csv")
MOUSE_DF = pd.read_csv(ibl_mouse_data_path)
mID = 'CSHL_003'
tmp = MOUSE_DF[MOUSE_DF['subject']=='CSHL_003'] 
# tmp.to_csv(ibl_data_path+os.sep+'CSHL_003'+"_ibl_processed.csv")


## for the Rat DATA - figure 5 and after
SPATH=  tpath(r'Talks\2022-01-13 - pillow\TestData')
akrami_rat_data_path = tpath(r"Talks\2022-01-13 - pillow\Brody_ratdata\rat_behavior.csv")
RAT_DF = pd.read_csv(akrami_rat_data_path)
RAT_DF = RAT_DF[RAT_DF["training_stage"] > 2]  # Remove trials from early training
RAT_DF = RAT_DF[~np.isnan(RAT_DF["choice"])]   # Remove mistrials


### from Rumbaughlab
### with prior single animal
#############################

mypath = tpath(r'Sheldon\All_WDIL\WDIL007_SyngapKO_high_stim_1step_12-16-19\forPsyTrack')
# mypath = tpath(r'Sheldon\All_WDIL\for psytrack\WDIL010Box1+2')

SPATH =  mypath+os.sep+'output'
os.makedirs(SPATH, exist_ok = True)

listFiles(mypath)

## load or generate the data
if glob.glob(mypath+os.sep+'allDat.csv') == [mypath+os.sep+'allDat.csv']:
    allDat = pd.read_csv(mypath+os.sep+'allDat.csv')
else:
    allDat = formatWDILfile(mypath)
    allDat.to_csv(mypath+os.sep+'allDat.csv')

## coding dat file
allDat = codingDatFile(allDat)

## add genotype info to the file
geno = pd.read_csv(tpath(r"Sheldon\All_WDIL\WDIL007_SyngapKO_high_stim_1step_12-16-19\animals.csv"))
allDat = pd.merge(allDat, geno, on ='sID')

## adding the genotype to the dat file
## then as a first pass can split the file and process wt or het

sID = 1753 #input the idname of the subeject
fname = SPATH+os.sep+str(sID)+'_fig5b_data.npz'
psyCompute(allDat, sID)



### from Rumbaughlab
### with prior single animal
#############################

# takes roughly 30 min 
all_id = allDat['sID'].unique()
a = time.time()
for i, sID in enumerate(all_id):
    print(i, sID)
    try:
        psyCompute(allDat, sID)
    except:
        print('error with: ', sID)
b = time.time()
print(b-a)


## create a list for the one that acutally worked 
## so this is based on the output not the initial list

for i in ['wt', 'het']:
    plotLabelsandW(geno=i, SPATH)