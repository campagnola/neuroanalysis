from __future__ import division
import numpy as np
from .data import Trace


def zero_crossing_events(data, min_length=3, min_peak=0.0, min_sum=0.0, noise_threshold=None):
    """Locate events of any shape in a signal. Works by finding regions of the signal
    that deviate from noise, and filtering with multiple criteria such as peak value,
    area under the event, and duration of the event.
    
    This algorithm relies on having data where the signal is lower frequency than the noise.
    Any low frequency noise must be removed by high-pass filtering to produce a flat baseline.
    In addition, the event amplitude must be sufficiently large that the signal does not 
    cross 0 within a single event (low-pass filtering may be necessary to achive this).
    
    Returns an array of events where each row is (start, length, sum, peak)
    """
    ## just make sure this is an ndarray and not a MetaArray before operating..
    #p = Profiler('findEvents')
    #p.mark('view')
    
    if isinstance(data, Trace):
        xvals = data.time_values
        data1 = data.data
    else:
        xvals = None
        data1 = data.view(np.ndarray)
    
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


def threshold_events(data, threshold, adjust_times=True, baseline=0.0):
    """Finds regions in a trace that cross a threshold value (as measured by distance from baseline). Returns the index, time, length, peak, and sum of each event.
    Optionally adjusts times to an extrapolated baseline-crossing."""
    threshold = abs(threshold)
    data1 = data.view(np.ndarray)
    data1 = data1 - baseline
    #if (hasattr(data, 'implements') and data.implements('MetaArray')):
    try:
        xvals = data.xvals(0)
        dt = xvals[1]-xvals[0]
    except:
        dt = 1
        xvals = None
    
    ## find all threshold crossings
    masks = [(data1 > threshold).astype(np.byte), (data1 < -threshold).astype(np.byte)]
    hits = []
    for mask in masks:
        diff = mask[1:] - mask[:-1]
        onTimes = np.argwhere(diff==1)[:,0]+1
        offTimes = np.argwhere(diff==-1)[:,0]+1
        #print mask
        #print diff
        #print onTimes, offTimes
        if len(onTimes) == 0 or len(offTimes) == 0:
            continue
        if offTimes[0] < onTimes[0]:
            offTimes = offTimes[1:]
            if len(offTimes) == 0:
                continue
        if offTimes[-1] < onTimes[-1]:
            onTimes = onTimes[:-1]
        for i in xrange(len(onTimes)):
            hits.append((onTimes[i], offTimes[i]))
    
    ## sort hits  ## NOTE: this can be sped up since we already know how to interleave the events..
    hits.sort(lambda a,b: cmp(a[0], b[0]))
    
    nEvents = len(hits)
    if xvals is None:
        events = np.empty(nEvents, dtype=[('index',int),('len', int),('sum', float),('peak', float),('peakIndex', int)])  ### rows are [start, length, sum]
    else:
        events = np.empty(nEvents, dtype=[('index',int),('time',float),('len', int),('sum', float),('peak', float),('peakIndex', int)])  ### rows are     

    mask = np.ones(nEvents, dtype=bool)
    
    ## Lots of work ahead:
    ## 1) compute length, peak, sum for each event
    ## 2) adjust event times if requested, then recompute parameters
    for i in range(nEvents):
        t1, t2 = hits[i]
        ln = t2-t1
        evData = data1[t1:t2]
        sum = evData.sum()
        if sum > 0:
            #peak = evData.max()
            #ind = argwhere(evData==peak)[0][0]+t1
            peakInd = np.argmax(evData)
        else:
            #peak = evData.min()
            #ind = argwhere(evData==peak)[0][0]+t1
            peakInd = np.argmin(evData)
        peak = evData[peakInd]
        peakInd += t1
            
        #print "event %f: %d" % (xvals[t1], t1) 
        if adjust_times:  ## Move start and end times outward, estimating the zero-crossing point for the event
        
            ## adjust t1 first
            mind = np.argmax(evData)
            pdiff = abs(peak - evData[0])
            if pdiff == 0:
                adj1 = 0
            else:
                adj1 = int(threshold * mind / pdiff)
                adj1 = min(ln, adj1)
            t1 -= adj1
            #print "   adjust t1", adj1
            
            ## check for collisions with previous events
            if i > 0:
                #lt2 = events[i-1]['index'] + events[i-1]['len']
                lt2 = hits[i-1][1]
                if t1 < lt2:
                    diff = lt2-t1   ## if events have collided, force them to compromise
                    tot = adj1 + lastAdj
                    if tot != 0:
                        d1 = diff * float(lastAdj) / tot
                        d2 = diff * float(adj1) / tot
                        #events[i-1]['len'] -= (d1+1)
                        hits[i-1] = (int(hits[i-1][0]), int(hits[i-1][1]-(d1+1)))
                        t1 += d2
                        #recompute[i-1] = True
                        #print "  correct t1", d2, "  correct prev.", d1+1
            #try:
                #print "   correct t1", d2, "  correct prev.", d1+1
            #except:
                #pass
            
            ## adjust t2
            mind = ln - mind
            pdiff = abs(peak - evData[-1])
            if pdiff == 0:
                adj2 = 0
            else:
                adj2 = int(threshold * mind / pdiff)
                adj2 = min(ln, adj2)
            t2 += adj2
            lastAdj = adj2
            #print "  adjust t2", adj2
            
            #recompute[i] = True
            
        #starts.append(t1)
        #stops.append(t2)
        hits[i] = (int(t1), int(t2))
        events[i]['peak'] = peak
        #if index == 'peak':
            #events[i]['index']=ind
        #else:
        events[i]['index'] = t1
        events[i]['peakIndex'] = peakInd
        events[i]['len'] = ln
        events[i]['sum'] = sum
        
    if adjust_times:  ## go back and re-compute event parameters.
        for i in range(nEvents):
            t1, t2 = hits[i]
            
            ln = t2-t1
            evData = data1[t1:t2]
            sum = evData.sum()
            if len(evData) == 0:
                mask[i] = False
                continue
            if sum > 0:
                #peak = evData.max()
                #ind = argwhere(evData==peak)[0][0]+t1
                peakInd = np.argmax(evData)
            else:
                #peak = evData.min()
                #ind = argwhere(evData==peak)[0][0]+t1
                peakInd = np.argmin(evData)
            peak = evData[peakInd]
            peakInd += t1
                
            events[i]['peak'] = peak
            #if index == 'peak':
                #events[i]['index']=ind
            #else:
            events[i]['index'] = t1
            events[i]['peakIndex'] = peakInd
            events[i]['len'] = ln
            events[i]['sum'] = sum
    
    ## remove masked events
    events = events[mask]
    
    if xvals is not None:
        events['time'] = xvals[events['index']]
        
    #for i in xrange(len(events)):
        #print events[i]['time'], events[i]['peak']

    return events


def rolling_sum(data, n):
    """A sliding-window filter that returns the sum of n source values for each
    value in the output array::
    
        output[i] = sum(input[i:i+n])
    
    """
    d1 = np.cumsum(data)
    d2 = np.empty(len(d1) - n + 1, dtype=data.dtype)
    d2[0] = d1[n-1]  # copy first point
    d2[1:] = d1[n:] - d1[:-n]  # subtract
    return d2
    

def clements_bekkers(data, template):
    """Sliding, scale-invariant template matching algorithm.

    Slides a template along a signal, measuring their similarity and scale difference
    at each sample. Similar to deconvolution.
    
    Parameters
    ----------
    data : array
        The signal data to process
    template : array
        A template 
    
    Returns
    -------
    detection_criterion : array
        Indicates how well the template matches the signal at each sample.
    scale : array
        The scale factor providing the best fit between template and signal at each sample.
    offset : array
        The y-offset of the best template fit at each sample.
    
    Notes
    -----
    Fast and sensitive for well-isolated events but performs poorly on overlapping events.
    See Clements & Bekkers, Biophysical Journal, 73: 220-229, 1997.
    """
    # Strip out meta-data for faster computation
    D = data.view(ndarray)
    T = template.view(ndarray)
    
    # Prepare a bunch of arrays we'll need later
    N = len(T)
    sumT = T.sum()
    sumT2 = (T**2).sum()
    sumD = rolling_sum(D, N)
    sumD2 = rolling_sum(D**2, N)
    sumTD = correlate(D, T, mode='valid')
    
    # compute scale factor, offset at each location:
    scale = (sumTD - sumT * sumD / N) / (sumT2 - sumT**2 / N)
    offset = (sumD - scale * sumT) / N
    
    # compute SSE at every location
    SSE = sumD2 + scale**2 * sumT2 + N * offset**2 - 2 * (scale*sumTD + offset*sumD - scale*offset*sumT)
    
    # finally, compute error and detection criterion
    error = sqrt(SSE / (N-1))
    DC = scale / error
    return DC, scale, offset


def exp_deconvolve(trace, tau):
    """Exponential deconvolution used to isolate overlapping events; works nicely on PSPs, calcium transients, etc.

    See: Richardson & Silberberg 2008, "Measurement and Analysis of Postsynaptic Potentials Using 
         a Novel Voltage-Deconvolution Method"
    """
    dt = trace.dt
    arr = trace.data
    deconv = arr[:-1] + (tau / dt) * (arr[1:] - arr[:-1])
    if trace.has_time_values:
        # data is one sample shorter; clip time values to match.
        return trace.copy(data=deconv, time_values=trace.time_values[:-1])
    else:
        return trace.copy(data=deconv)
    
def exp_reconvolve(trace, tau):
    # equivalent to subtracting an exponential decay from the original unconvolved signal
    dt = trace.dt
    d = np.zeros(trace.data.shape, trace.data.dtype)
    dtt = dt / tau
    dtti = 1. - dtt
    for i in range(1, len(d)):
        d[i] = dtti * d[i-1] + dtt * trace.data[i-1]
    return trace.copy(data=d)
