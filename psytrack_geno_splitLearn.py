def plotLabelsandW(geno, SPATH, splitCriteria = False):
    '''
    this function will output all the labels and weight for a givien genotype
    based on the corresponding files and select the genotype of interest

    If splitCriteria is set to True then the 'animals.csv' file must have a column named 'criteria' with pass=1 and fail=0

    args:
    corresp(pd.DataFrame): data frame with file path sID and geno
    geno(str): geno of intrest either 'wt' or 'het'
    splitCriteria(bool) set to True to split the given genotype by criteria
    '''

    ## have the corresponding file generatede 
    npzFiles = glob.glob(SPATH+os.sep+'*.npz')
    corresp = pd.DataFrame({'fname':npzFiles})
    corresp['sID'] = corresp['fname'].str.split(os.sep).str[-1].str.split('_').str[0].astype(int)

    ## check and implement the genotypes
    # try: geno
    # except: geno = None
    # if geno is None:
    animals = pd.read_csv(os.sep.join(SPATH.split(os.sep)[:-2])+os.sep+'animals.csv')

    corresp = pd.merge(corresp, animals, on='sID')

    ## 
    cWTorHet = corresp[corresp['geno'] == geno]

    ## save all the data for the given genotype
    if (splitCriteria == False):
    
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

    ##Split the data for a genotype between animals that reached criteria and those that did not. Save each separately
    else:
        

        criteria = cWTorHet[cWTorHet['criteria'] == 1]
        notCriteria = cWTorHet[cWTorHet['criteria'] == 0]

        criteria_labels = []
        criteria_w = []

        notCriteria_labels = []
        notCriteria_w = []

        ##fill labels and weights for animals which reached criteria of the given genotype
        for i,j in criteria.iterrows():
            print(i,j)
            rat = np.load(j['fname'], allow_pickle=True)['dat'].item()
            
            labels = []
            for j in sorted(rat['weights'].keys()):
                labels += [j]*rat['weights'][j]
                
            criteria_labels += [np.array(labels)]
            criteria_w += [rat['wMode']] 

        ##fill labels and weights for animals which did not reached criteria of the given genotype
        for i,j in notCriteria.iterrows():
            print(i,j)
            rat = np.load(j['fname'], allow_pickle=True)['dat'].item()
            
            labels = []
            for j in sorted(rat['weights'].keys()):
                labels += [j]*rat['weights'][j]
                
            notCriteria_labels += [np.array(labels)]
            notCriteria_w += [rat['wMode']] 

##plot and save figure for animals of the genotype which reached criteria as Fig6a,b,d,e

        myFigsize = (3.6,1.8)
        plot_all(criteria_labels, criteria_w, ["s1"], myFigsize)
        plt.ylim(-1, 15)
        # plt.subplots_adjust(0,0,1,1) 
        # plt.gca().set_yticks([-2,0,2])
        # plt.gca().set_xticklabels([])
        plt.savefig(SPATH +os.sep+ geno+ "Fig7a.pdf")

        plot_all(criteria_labels, criteria_w, ["bias"], myFigsize)
        plt.ylim(-7, 2)
        # plt.gca().set_yticks([-2,0,2])
        # plt.gca().set_xticklabels([])
        # plt.gca().set_yticklabels([])
        # plt.subplots_adjust(0,0,1,1) 
        plt.savefig(SPATH +os.sep+ geno+ "Fig7b.pdf")

        plot_all(criteria_labels, criteria_w, ["h"], myFigsize)
        plt.ylim(-1, 1)
        # # plt.gca().set_yticklabels([])
        # plt.subplots_adjust(0,0,1,1) 
        plt.savefig(SPATH +os.sep+ geno+ "Fig7d.pdf")


        plot_all(criteria_labels, criteria_w, ["c"], myFigsize)
        plt.ylim(-1, 1)
        # plt.gca().set_yticklabels([])
        # plt.gca().set_xticklabels([])
        # plt.subplots_adjust(0,0,1,1) 
        plt.savefig(SPATH +os.sep+ geno+ "Fig7e.pdf")

##plot and save figure for animals of the genotype which did not reached criteria as Fig6f,g,i,j

        myFigsize = (3.6,1.8)
        plot_all(notCriteria_labels, notCriteria_w, ["s1"], myFigsize)
        plt.ylim(-1, 15)
        # plt.subplots_adjust(0,0,1,1) 
        # plt.gca().set_yticks([-2,0,2])
        # plt.gca().set_xticklabels([])
        plt.savefig(SPATH +os.sep+ geno+ "Fig7f.pdf")

        plot_all(notCriteria_labels, notCriteria_w, ["bias"], myFigsize)
        plt.ylim(-7, 2)
        # plt.gca().set_yticks([-2,0,2])
        # plt.gca().set_xticklabels([])
        # plt.gca().set_yticklabels([])
        # plt.subplots_adjust(0,0,1,1) 
        plt.savefig(SPATH +os.sep+ geno+ "Fig7g.pdf")

        plot_all(notCriteria_labels, notCriteria_w, ["h"], myFigsize)
        plt.ylim(-1, 1)
        # # plt.gca().set_yticklabels([])
        # plt.subplots_adjust(0,0,1,1) 
        plt.savefig(SPATH +os.sep+ geno+ "Fig7i.pdf")


        plot_all(notCriteria_labels, notCriteria_w, ["c"], myFigsize)
        plt.ylim(-1, 1)
        # plt.gca().set_yticklabels([])
        # plt.gca().set_xticklabels([])
        # plt.subplots_adjust(0,0,1,1) 
        plt.savefig(SPATH +os.sep+ geno+ "Fig7j.pdf")