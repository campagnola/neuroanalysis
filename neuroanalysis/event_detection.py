import numpy as np


def zero_crossing_events(data, min_length=3, min_peak=0.0, min_sum=0.0, noise_threshold=None):
    """Locate events of any shape in a signal. Works by finding regions of the signal
    that deviate from noise, using the area beneath the deviation as the detection criteria.
    
    Makes the following assumptions about the signal:
      - noise is gaussian
      - baseline is centered at 0 (high-pass filtering may be required to achieve this).
      - no 0 crossings within an event due to noise (low-pass filtering may be required to achieve this)
      - Events last more than min_length samples
      Return an array of events where each row is (start, length, sum, peak)
    """
    ## just make sure this is an ndarray and not a MetaArray before operating..
    #p = Profiler('findEvents')
    data1 = data.view(np.ndarray)
    #p.mark('view')
    xvals = None
    if (hasattr(data, 'implements') and data.implements('MetaArray')):
        try:
            xvals = data.xvals(0)
        except:
            pass
    
    
    ## find all 0 crossings
    mask = data1 > 0
    diff = mask[1:] - mask[:-1]  ## mask is True every time the trace crosses 0 between i and i+1
    times1 = np.argwhere(diff)[:, 0]  ## index of each point immediately before crossing.
    
    times = np.empty(len(times1)+2, dtype=times1.dtype)  ## add first/last indexes to list of crossing times
    times[0] = 0                                         ## this is a bit suspicious, but we'd rather know
    times[-1] = len(data1)                               ## about large events at the beginning/end
    times[1:-1] = times1                                 ## rather than ignore them.
    #p.mark('find crossings')
    
    ## select only events longer than min_length.
    ## We do this check early for performance--it eliminates the vast majority of events
    longEvents = np.argwhere(times[1:] - times[:-1] > min_length)
    if len(longEvents) < 1:
        nEvents = 0
    else:
        longEvents = longEvents[:, 0]
        nEvents = len(longEvents)
    
    ## Measure sum of values within each region between crossings, combine into single array
    if xvals is None:
        events = np.empty(nEvents, dtype=[('index',int),('len', int),('sum', float),('peak', float)])  ### rows are [start, length, sum]
    else:
        events = np.empty(nEvents, dtype=[('index',int),('time',float),('len', int),('sum', float),('peak', float)])  ### rows are [start, length, sum]
    #p.mark('empty %d -> %d'% (len(times), nEvents))
    #n = 0
    for i in range(nEvents):
        t1 = times[longEvents[i]]+1
        t2 = times[longEvents[i]+1]+1
        events[i]['index'] = t1
        events[i]['len'] = t2-t1
        evData = data1[t1:t2]
        events[i]['sum'] = evData.sum()
        if events[i]['sum'] > 0:
            peak = evData.max()
        else:
            peak = evData.min()
        events[i]['peak'] = peak
    #p.mark('generate event array')
    
    if xvals is not None:
        events['time'] = xvals[events['index']]
    
    if noise_threshold > 0:
        ## Fit gaussian to peak in size histogram, use fit sigma as criteria for noise rejection
        stdev = measureNoise(data1)
        #p.mark('measureNoise')
        hist = histogram(events['sum'], bins=100)
        #p.mark('histogram')
        histx = 0.5*(hist[1][1:] + hist[1][:-1]) ## get x values from middle of histogram bins
        #p.mark('histx')
        fit = fitGaussian(histx, hist[0], [hist[0].max(), 0, stdev*3, 0])
        #p.mark('fit')
        sigma = fit[0][2]
        minSize = sigma * noise_threshold
        
        ## Generate new set of events, ignoring those with sum < minSize
        #mask = abs(events['sum'] / events['len']) >= minSize
        mask = abs(events['sum']) >= minSize
        #p.mark('mask')
        events = events[mask]
        #p.mark('select')

    if min_peak > 0:
        events = events[abs(events['peak']) > min_peak]
    
    if min_sum > 0:
        events = events[abs(events['sum']) > min_sum]
    
        
    return events

