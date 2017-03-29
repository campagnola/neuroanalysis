"""
Description:

    Experimetns that characterize the functional synaptic connectivity between
    two neurons often rely on being able to evoke a spike in the presynaptic
    cell and detect an evoked synaptic response in the postsynaptic cell. 
    These synaptic responses can be difficult to distinguish from the constant
    background of spontaneous synaptic activity.

    The method implemented here assumes that spontaneous activity can be
    described by a poisson process. For any given series of synaptic events, 
    we calculate the probability that the event times could have been generated
    by a poisson process. 
    
    
    (1) Obvious, immediate rate change
    
    |___||____|_|_______|____|___|___|_|||||||_|_||__|_||___|____|__|_|___|____|___
                                      ^
    (2) Obvious, delayed rate change
    
    |___||____|_|_______|____|___|___|_|____|__|_|___|___|_||_|_||_|_|__|_|_||_____
                                      ^
    (3) Non-obvious rate change, but responses have good precision   
    
    |______|______|_________|_______|____|______|________|________|_________|______
    _____|___________|_______|___|_______|__________|____|______|______|___________
    ___|________|_________|___|______|___|__|_________|____|_______|___________|___
                                      ^
    (4) Very low spont rate (cannot measure intervals between events)
        with good response precision
        
    ______________________________________|________________________________________
    ________|___________________________________|__________________________________
    _________________________________________|________________________|____________
                                      ^
    (5) Non-obvious rate change, but response amplitudes are very different
    
    __,______.___,_______.___,_______,_____|___,_____._________,_,______.______,___
                                      ^

    

"""


import numpy as np
import scipy
import scipy.stats as stats
import scipy.misc
import scipy.interpolate
import acq4.pyqtgraph as pg
import acq4.pyqtgraph.console
import user
import acq4.pyqtgraph.multiprocess as mp
import os


def poissonProcess(rate, tmax=None, n=None):
    """Simulate a poisson process; return a list of event times"""
    events = []
    t = 0
    while True:
        t += np.random.exponential(1./rate)
        if tmax is not None and t > tmax:
            break
        events.append(t)
        if n is not None and len(events) >= n:
            break
    return np.array(events)


def poissonProb(n, t, l, clip=False):
    """
    For a poisson process, return the probability of seeing at least *n* events in *t* seconds given
    that the process has a mean rate *l*.
    """
    if l == 0:
        if np.isscalar(n):
            if n == 0:
                return 1.0
            else:
                return 1e-25
        else:
            return np.where(n==0, 1.0, 1e-25)
    
    p = stats.poisson(l*t).sf(n)   
    if clip:
        p = np.clip(p, 0, 1.0-1e-25)
    return p


def gaussProb(amps, mean, stdev):
    ## Return the survival function for gaussian distribution 
    if len(amps) == 0:
        return 1.0
    return stats.norm(mean, stdev).sf(amps)
    
    
class PoissonScore:
    """
    Class for computing a statistic that asks "what is the probability that a poisson process
    would generate a set of events like this"
    
    General procedure:
      1. For each event n in a list of events, compute the probability of a poisson
         process generating at least n-1 events in the time up to event n (this is 
         poissonProb() applied individually to each event)
      2. The maximum value over all events is the score. For multiple trials, simply
         mix together all events and assume an accordingly faster poisson process.
      3. Normalize the score to a probability using a precomputed table generated
         by a poisson process simulations.
    """
    
    
    normalizationTable = None
    
        
    @classmethod
    def score(cls, ev, rate, tMax=None, normalize=True, **kwds):
        """
        Compute poisson score for a set of events.
        ev must be a list of record arrays. Each array describes a set of events; only required field is 'time'
        *rate* may be either a single value or a list (in which case the mean will be used)
        """
        nSets = len(ev)
        events = np.concatenate(ev)

        if not np.isscalar(rate):   ### Is this valid???  I think so..
            rate = np.mean(rate)
        
        if len(events) == 0:
            score = 1.0
        else:
            #ev = [x['time'] for x in ev]  ## select times from event set
            #ev = np.concatenate(ev)   ## mix events together
            ev = events['time']
            
            nVals = np.array([(ev<=t).sum()-1 for t in ev]) ## looks like arange, but consider what happens if two events occur at the same time.
            pi = poissonProb(nVals, ev, rate*nSets)  ## note that by using n=0 to len(ev)-1, we correct for the fact that the time window always ends at the last event
            pi = 1.0 / pi
            
            ## apply extra score for uncommonly large amplitudes
            ## (note: by default this has no effect; see amplitudeScore)
            ampScore = cls.amplitudeScore(events, **kwds)
            pi *= ampScore
            
            mp = pi.max()
            #mpp = min(cls.maxPoissonProb(ev, rate*nSets), 1.0-1e-12)  ## don't allow returning inf
            #mpp = min(mp, 1.0-1e-12)
            
            score = mp
            #score =  1.0 / (1.0 - mpp)
            
            
        #n = len(ev)
        if normalize:
            ret = cls.mapScore(score, rate*tMax*nSets)
        else:
            ret = score
        if np.isscalar(ret):
            assert not np.isnan(ret)
        else:
            assert not any(np.isnan(ret))
        
        return ret

    @classmethod
    def amplitudeScore(cls, events, **kwds):
        """Computes extra probability information about events based on their amplitude.
        Inputs to this method are:
            events: record array of events; fields include 'time' and 'amp'
            
        By default, no extra score is applied for amplitude (but see also PoissonRepeatAmpScore)
        """
        return np.ones(len(events))

    @classmethod
    def mapScore(cls, x, n):
        """
        Map score x to probability given we expect n events per set
        """
        if cls.normalizationTable is None:
            cls.normalizationTable = cls.generateNormalizationTable()
            cls.extrapolateNormTable()
            
        nind = max(0, np.log(n)/np.log(2))
        n1 = np.clip(int(np.floor(nind)), 0, cls.normalizationTable.shape[1]-2)
        n2 = n1+1
        
        mapped1 = []
        for i in [n1, n2]:
            norm = cls.normalizationTable[:,i]
            ind = np.argwhere(norm[0] > x)
            if len(ind) == 0:
                ind = len(norm[0])-1
            else:
                ind = ind[0,0]
            if ind == 0:
                ind = 1
            x1, x2 = norm[0, ind-1:ind+1]
            y1, y2 = norm[1, ind-1:ind+1]
            if x1 == x2:
                s = 0.0
            else:
                s = (x-x1) / float(x2-x1)
            mapped1.append(y1 + s*(y2-y1))
        
        mapped = mapped1[0] + (mapped1[1]-mapped1[0]) * (nind-n1)/float(n2-n1)
        
        ## doesn't handle points outside of the original data.
        #mapped = scipy.interpolate.griddata(poissonScoreNorm[0], poissonScoreNorm[1], [x], method='cubic')[0]
        #normTable, tVals, xVals = poissonScoreNorm
        #spline = scipy.interpolate.RectBivariateSpline(tVals, xVals, normTable)
        #mapped = spline.ev(n, x)[0]
        #raise Exception()
        assert not (np.isinf(mapped) or np.isnan(mapped))
        assert mapped>0
        return mapped

    @classmethod
    def generateRandom(cls, rate, tMax, reps=3):
        if np.isscalar(rate):
            rate = [rate]*reps
        ret = []
        for i in range(reps):
            times = poissonProcess(rate[i], tMax)
            ev = np.empty(len(times), dtype=[('time', float), ('amp', float)])
            ev['time'] = times
            ev['amp'] = np.random.normal(size=len(times))
            ret.append(ev)
        return ret
        
    @classmethod
    def generateNormalizationTable(cls, nEvents=1000000):
        ## table looks like this:
        ##   (2 x M x N)
        ##   Axis 0:  (score, mapped)
        ##   Axis 1:  expected number of events  [1, 2, 4, 8, ...]
        ##   Axis 2:  score axis 
        
        ## To map:
        ##    determine axis-1 index by expected number of events
        ##    look up axis-2 index from table[0, ind1]
        ##    look up mapped score at table[1, ind1, ind2]
        
        
        ## parameters determining sample space for normalization table
        rate = 1.0
        tVals = 2**np.arange(9)  ## set of tMax values
        nev = (nEvents / (rate*tVals)**0.5).astype(int)  # number of events to generate for each tMax value
        
        xSteps = 1000
        r = 10**(30./xSteps)
        xVals = r ** np.arange(xSteps)  ## log spacing from 1 to 10**20 in 500 steps
        tableShape = (2, len(tVals), len(xVals))
        
        path = os.path.dirname(__file__)
        cacheFile = os.path.join(path, '%s_normTable_%s_float64.dat' % (cls.__name__, 'x'.join(map(str,tableShape))))
        
        if os.path.exists(cacheFile):
            norm = np.fromstring(open(cacheFile).read(), dtype=np.float64).reshape(tableShape)
        else:
            print "Generating poisson score normalization table (will be cached here: %s)" % cacheFile
            norm = np.empty(tableShape)
            counts = []
            with mp.Parallelize(counts=counts) as tasker:
                for task in tasker:
                    count = np.zeros(tableShape[1:], dtype=float)
                    for i, t in enumerate(tVals):
                        n = nev[i] / tasker.numWorkers()
                        for j in xrange(int(n)):
                            if j%10000==0:
                                print "%d/%d  %d/%d" % (i, len(tVals), j, int(n))
                                tasker.process()
                            ev = cls.generateRandom(rate=rate, tMax=t, reps=1)
                            
                            score = cls.score(ev, rate, normalize=False)
                            ind = np.log(score) / np.log(r)
                            count[i, :int(ind)+1] += 1
                    tasker.counts.append(count)
                            
            count = sum(counts)
            count[count==0] = 1
            norm[0] = xVals.reshape(1, len(xVals))
            norm[1] = nev.reshape(len(nev), 1) / count
            
            open(cacheFile, 'wb').write(norm.tostring())
        
        return norm
        
    @classmethod
    def testMapping(cls, rate=1.0, tMax=1.0, n=10000, reps=3):
        scores = np.empty(n)
        mapped = np.empty(n)
        ev = []
        for i in xrange(len(scores)):
            ev.append(cls.generateRandom(rate, tMax, reps))
            scores[i] = cls.score(ev[-1], rate, tMax=tMax, normalize=False)
            mapped[i] = cls.mapScore(scores[i], np.mean(rate)*tMax*reps)
        
        for j in [1,2,3,4]:
            print "  %d: %f" % (10**j, (mapped>10**j).sum() / float(n))
        return ev, scores, mapped
        
    @classmethod
    def showMap(cls):
        plt = pg.plot()
        for i in range(cls.normalizationTable.shape[1]):
            plt.plot(cls.normalizationTable[0,i], cls.normalizationTable[1,i], pen=(i, 14), symbolPen=(i,14), symbol='o')
    
    @classmethod
    def poissonScoreBlame(ev, rate):
        nVals = np.array([(ev<=t).sum()-1 for t in ev]) 
        pp1 = 1.0 /   (1.0 - cls.poissonProb(nVals, ev, rate, clip=True))
        pp2 = 1.0 /   (1.0 - cls.poissonProb(nVals-1, ev, rate, clip=True))
        diff = pp1 / pp2
        blame = np.array([diff[np.argwhere(ev >= ev[i])].max() for i in range(len(ev))])
        return blame

    @classmethod
    def extrapolateNormTable(cls):
        ## It appears that, on a log-log scale, the normalization curves appear to become linear after reaching
        ## about 50 on the y-axis. 
        ## we can use this to overwrite all the junk at the end caused by running too few test iterations.
        d = cls.normalizationTable
        for n in range(d.shape[1]):
            trace = d[:,n]
            logtrace = np.log(trace)
            ind1 = np.argwhere(trace[1] > 60)[0,0]
            ind2 = np.argwhere(trace[1] > 100)[0,0]
            dd = logtrace[:,ind2] - logtrace[:,ind1]
            slope = dd[1]/dd[0]
            npts = trace.shape[1]-ind2
            yoff = logtrace[1,ind2] - logtrace[0,ind2] * slope
            trace[1,ind2:] = np.exp(logtrace[0,ind2:] * slope + yoff)
        
        

class PoissonAmpScore(PoissonScore):
    
    normalizationTable = None
    
    @classmethod
    def amplitudeScore(cls, events, ampMean=1.0, ampStdev=1.0, **kwds):
        """Computes extra probability information about events based on their amplitude.
        Inputs to this method are:
            events: record array of events; fields include 'time' and 'amp'
            times:  the time points at which to compute probability values
                    (the output must have the same length)
            ampMean, ampStdev: population statistics of spontaneous events
        """
        if ampStdev == 0.0:    ## no stdev information; cannot determine probability.
            return np.ones(len(events))
        scores = 1.0 / np.clip(gaussProb(events['amp'], ampMean, ampStdev), 1e-100, np.inf)
        assert(not np.any(np.isnan(scores) | np.isinf(scores)))
        return scores



if __name__ == '__main__':
    import pyqtgraph as pg
    import pyqtgraph.console
    app = pg.mkQApp()
    con = pg.console.ConsoleWidget()
    con.show()
    con.catchAllExceptions()



    ## Create a set of test cases:

    reps = 3
    trials = 30
    spontRate = [2., 3., 5.]
    miniAmp = 1.0
    tMax = 0.5

    def randAmp(n=1, quanta=1):
        return np.random.gamma(4., size=n) * miniAmp * quanta / 4.

    ## create a standard set of spontaneous events
    spont = [] ## trial, rep
    allAmps = []
    for i in range(trials):
        spont.append([])
        for j in range(reps):
            times = poissonProcess(spontRate[j], tMax)
            amps = randAmp(len(times))  ## using scale=4 gives a nice not-quite-gaussian distribution
            source = ['spont'] * len(times)
            spont[i].append((times, amps, source))
            allAmps.append(amps)
            
    miniStdev = np.concatenate(allAmps).std()


    def spontCopy(i, j, extra):
        times, amps, source = spont[i][j]
        ev = np.zeros(len(times)+extra, dtype=[('time', float), ('amp', float), ('source', object)])
        ev['time'][:len(times)] = times
        ev['amp'][:len(times)] = amps
        ev['source'][:len(times)] = source
        return ev
        
    ## copy spont. events and add on evoked events
    testNames = []
    tests = [[[] for i in range(trials)] for k in range(7)]  # test, trial, rep
    for i in range(trials):
        for j in range(reps):
            ## Test 0: no evoked events
            testNames.append('No evoked')
            tests[0][i].append(spontCopy(i, j, 0))

            ## Test 1: 1 extra event, single quantum, short latency
            testNames.append('1ev, fast')
            ev = spontCopy(i, j, 1)
            ev[-1] = (np.random.gamma(1.0) * 0.01, 1, 'evoked')
            tests[1][i].append(ev)

            ## Test 2: 2 extra events, single quantum, short latency
            testNames.append('2ev, fast')
            ev = spontCopy(i, j, 2)
            for k, t in enumerate(np.random.gamma(1.0, size=2)*0.01):
                ev[-(k+1)] = (t, 1, 'evoked')
            tests[2][i].append(ev)

            ## Test 3: 3 extra events, single quantum, long latency
            testNames.append('3ev, slow')
            ev = spontCopy(i, j, 3)
            for k,t in enumerate(np.random.gamma(1.0, size=3)*0.07):
                ev[-(k+1)] = (t, 1, 'evoked')
            tests[3][i].append(ev)

            ## Test 4: 1 extra event, 2 quanta, short latency
            testNames.append('1ev, 2x, fast')
            ev = spontCopy(i, j, 1)
            ev[-1] = (np.random.gamma(1.0)*0.01, 2, 'evoked')
            tests[4][i].append(ev)

            ## Test 5: 1 extra event, 3 quanta, long latency
            testNames.append('1ev, 3x, slow')
            ev = spontCopy(i, j, 1)
            ev[-1] = (np.random.gamma(1.0)*0.05, 3, 'evoked')
            tests[5][i].append(ev)

            ## Test 6: 1 extra events specific time (tests handling of simultaneous events)
            #testNames.append('3ev simultaneous')
            #ev = spontCopy(i, j, 1)
            #ev[-1] = (0.01, 1, 'evoked')
            #tests[6][i].append(ev)
            
            ## 2 events, 1 failure
            testNames.append('0ev; 1ev; 2ev')
            ev = spontCopy(i, j, j)
            if j > 0:
                for k, t in enumerate(np.random.gamma(1.0, size=j)*0.01):
                    ev[-(k+1)] = (t, 1, 'evoked')
            tests[6][i].append(ev)
            

    #raise Exception()

    ## Analyze and plot all:

    def checkScores(scores):
        best = None
        bestn = None
        bestval = None
        for i in [0,1]:
            for j in range(scores.shape[1]): 
                x = scores[i,j]
                fn = (scores[0] < x).sum()
                fp = (scores[1] >= x).sum()
                diff = abs(fp-fn)
                if bestval is None or diff < bestval:
                    bestval = diff
                    best = x
                    bestn = (fp+fn)/2.
        return best, bestn
        
        
    algorithms = [
        ('Poisson Score', PoissonScore.score),
        ('Poisson Score + Amp', PoissonAmpScore.score),
        #('Poisson Multi', PoissonRepeatScore.score),
        #('Poisson Multi + Amp', PoissonRepeatAmpScore.score),
    ]

    win = pg.GraphicsWindow(border=0.3)
    with pg.ProgressDialog('processing..', maximum=len(tests)) as dlg:
        for i in range(len(tests)):
            first = (i == 0)
            last = (i == len(tests)-1)
            
            if first:
                evLabel = win.addLabel('Event amplitude', angle=-90, rowspan=len(tests))
            evPlt = win.addPlot()
            
            plots = []
            for title, fn in algorithms:
                if first:
                    label = win.addLabel(title, angle=-90, rowspan=len(tests))
                plt = win.addPlot()
                plots.append(plt)
                if first:
                    plt.register(title)
                else:
                    plt.setXLink(title)
                plt.setLogMode(False, True)
                plt.hideAxis('bottom')
                if last:
                    plt.showAxis('bottom')
                    plt.setLabel('bottom', 'Trial')
                    
                
            if first:
                evPlt.register('EventPlot1')
            else:
                evPlt.setXLink('EventPlot1')
            
            evPlt.hideAxis('bottom')
            evPlt.setLabel('left', testNames[i])
            if last:
                evPlt.showAxis('bottom')
                evPlt.setLabel('bottom', 'Event time', 's')
            
            trials = tests[i]
            scores = np.empty((len(algorithms), 2, len(trials)))
            repScores = np.empty((2, len(trials)))
            for j in range(len(trials)):
                
                ## combine all trials together for poissonScore tests
                ev = tests[i][j]
                spont = tests[0][j]
                evTimes = [x['time'] for x in ev]
                spontTimes = [x['time'] for x in spont]
                
                allEv = np.concatenate(ev)
                allSpont = np.concatenate(spont)
                
                colors = [pg.mkBrush(0,255,0,50) if source=='spont' else pg.mkBrush(255,255,255,50) for source in allEv['source']]
                evPlt.plot(x=allEv['time'], y=allEv['amp'], pen=None, symbolBrush=colors, symbol='d', symbolSize=8, symbolPen=None)
                
                for k, opts in enumerate(algorithms):
                    title, fn = opts
                    score1 = fn(ev, spontRate, tMax, ampMean=miniAmp, ampStdev=miniStdev)
                    score2 = fn(spont, spontRate, tMax, ampMean=miniAmp, ampStdev=miniStdev)
                    scores[k, :, j] = score1, score2
                    plots[k].plot(x=[j], y=[score1], pen=None, symbolPen=None, symbol='o', symbolBrush=(255,255,255,50))
                    plots[k].plot(x=[j], y=[score2], pen=None, symbolPen=None, symbol='o', symbolBrush=(0,255,0,50))

            
            ## Report on ability of each algorithm to separate spontaneous from evoked
            for k, opts in enumerate(algorithms):
                thresh, errors = checkScores(scores[k])
                plots[k].setTitle("%0.2g, %d" % (thresh, errors))
            
            ## Plot score histograms
            #bins = np.linspace(-1, 6, 50)
            #h1 = np.histogram(np.log10(scores[0, :]), bins=bins)
            #h2 = np.histogram(np.log10(scores[1, :]), bins=bins)
            #scorePlt.plot(x=0.5*(h1[1][1:]+h1[1][:-1]), y=h1[0], pen='w')
            #scorePlt.plot(x=0.5*(h2[1][1:]+h2[1][:-1]), y=h2[0], pen='g')
                
            #bins = np.linspace(-1, 14, 50)
            #h1 = np.histogram(np.log10(repScores[0, :]), bins=bins)
            #h2 = np.histogram(np.log10(repScores[1, :]), bins=bins)
            #repScorePlt.plot(x=0.5*(h1[1][1:]+h1[1][:-1]), y=h1[0], pen='w')
            #repScorePlt.plot(x=0.5*(h2[1][1:]+h2[1][:-1]), y=h2[0], pen='g')
                
            dlg += 1
            if dlg.wasCanceled():
                break
                
            win.nextRow()
        
