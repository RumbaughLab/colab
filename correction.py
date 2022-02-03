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

    # Calculate previous average tone value
    s_avg = df['Go/NoGo'][:-1]
    s_avg = (s_avg - np.mean(s_avg))/np.std(s_avg)
    s_avg = np.hstack(([0], s_avg))
    s_avg = s_avg * prior  # for trials without a valid previous trial, set to 0

    # Calculate previous correct answer
    h = (df['Go/NoGo'][:-1] * 2 - 1).astype(int)   # map from (0,1) to (-1,1)
    h = np.hstack(([0], h))
    h = h * prior  # for trials without a valid previous trial, set to 0
    
    # Calculate previous choice
    c = (df["choice"][:-1] * 2 - 1).astype(int)   # map from (0,1) to (-1,1)
    c = np.hstack(([0], c))
    c = c * prior  # for trials without a valid previous trial, set to 0
    
    ## note here that it could be useful to have different stimulus values 
    ## important to respect the dictionary psy.COLORS hence the name of the specific names of the inputs
    inputs = dict(s1 = np.array(df['Go/NoGo'])[:, None],
                  s_avg = np.array(s_avg)[:, None], 
                  h = np.array(h)[:, None],
                  c = np.array(c)[:, None])

    dat = dict(
        subject = subject,
        inputs = inputs,
        s1 = np.array(df['Go/NoGo']), # correspond to the go/noGo stim
        correct = np.array(df['Correct?']), # hit correspond to hit and correct rejection
        answer = np.array(df['Go/NoGo']), #this is the answer 
        y = np.array(df['choice']), #this correspond to the Lick
        dayLength=np.array(df.groupby(['session']).size()),
    )

    return dat