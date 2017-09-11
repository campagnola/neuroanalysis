# -*- coding: utf-8 -*-
"""
Simple neuron simulator for Python.
Also simulates voltage clamp and current clamp with access resistance.

Luke Campagnola 2015
"""

from collections import OrderedDict
from .sim import SimObject
from ..units import pF, mV, uF, cm


class Mechanism(SimObject):
    """Base class for simulation objects that interact with a section's
    membrane--channels, electrodes, etc.
    """
    def __init__(self, init_state, section=None, **kwds):
        SimObject.__init__(self, init_state, **kwds)
        self._name = kwds.pop('name', None)  # overwrite auto-generated name
        self._section = section
        self.dep_state_vars['I'] = self.current
        
    def current(self, state):
        """Return the membrane current being passed by this mechanism.
        
        Must be implemented in subclasses.
        """
        raise NotImplementedError()

    @property
    def name(self):
        if self._name is None:
            # pick a name that is unique to the section we live in
            
            # first collect all names
            names = []
            if self._section is None:
                return None
            for o in self._section.mechanisms:
                if isinstance(o, Mechanism) and o._name is None:
                    # skip to avoid recursion
                    continue
                names.append(o.name)
                
            # iterate until we find an unused name
            pfx = self._section.name + '.'
            name = pfx + self.type
            i = 1
            while name in names:
                name = pfx + self.type + str(i)
                i += 1
            self._name = name
        return self._name

    @property
    def section(self):
        return self._section
    
    @property
    def sim(self):
        return self.section.sim


class Channel(Mechanism):
    """Base class for simple ion channels.
    """
    # precomputed rate constant tables
    rates = None
    
    # maximum open probability (to be redefined by subclasses)
    max_op = 1.0
    
    @classmethod
    def compute_rates(cls):
        return
        
    def __init__(self, gmax=None, gbar=None, init_state=None, **kwds):
        Mechanism.__init__(self, init_state, **kwds)
        self._gmax = gmax
        self._gbar = gbar
            
        if self.rates is None:
            type(self).compute_rates()
        self.dep_state_vars['G'] = self.conductance
        self.dep_state_vars['OP'] = self.open_probability

    @property
    def gmax(self):
        if self._gmax is not None:
            return self._gmax
        else:
            return self._gbar * self.section.area
            
    @gmax.setter
    def gmax(self, v):
        self._gmax = v
        self._gbar = None
        
    @property
    def gbar(self):
        if self._gbar is not None:
            return self._gbar
        else:
            return self._gmax / self.section.area
        
    @gbar.setter
    def gbar(self, v):
        self._gbar = v
        self._gmax = None

    def conductance(self, state):
        op = self.open_probability(state)
        return self.gmax * op

    def current(self, state):
        vm = state[self.section, 'V']
        g = self.conductance(state)
        return -g * (vm - self.erev)

    @staticmethod
    def interpolate_rates(rates, val, minval, step):
        """Helper function for interpolating kinetic rates from precomputed
        tables.
        """
        i = (val - minval) / step
        i1 = int(i)
        i2 = i1 + 1
        s = i2 - i
        if i1 < 0:
            return rates[0]
        elif i2 >= len(rates):
            return rates[-1]
        else:
            return rates[i1] * s + rates[i2] * (1-s)


class Section(SimObject):
    type = 'section'
    
    def __init__(self, radius=None, cap=10*pF, vm=-65*mV, **kwds):
        self.cap_bar = 1 * uF/cm**2
        if radius is None:
            self.cap = cap
            self.area = cap / self.cap_bar
        else:
            self.area = 4 * 3.1415926 * radius**2
            self.cap = self.area * self.cap_bar
        self.ek = -77*mV
        self.ena = 50*mV
        self.ecl = -70*mV
        init_state = OrderedDict([('V', vm)])
        SimObject.__init__(self, init_state, **kwds)
        self.dep_state_vars['I'] = self.current
        self.mechanisms = []

    def add(self, mech):
        assert mech._section is None
        mech._section = self
        self.mechanisms.append(mech)
        self._sub_objs.append(mech)
        return mech

    def derivatives(self, state):
        Im = 0
        for mech in self.mechanisms:
            if not mech.enabled:
                continue
            Im += mech.current(state)
            
        dv = Im / self.cap
        return [dv]
    
    def current(self, state):
        """Return the current flowing across the membrane capacitance.
        """
        dv = self.derivatives(state)[0]
        return - self.cap * dv

    def conductance(self, state):
        """Return the total conductance of all channels in this section.
        
        This is for introspection; not used by the integrator.
        """
        g = 0
        for mech in self.mechanisms:
            if not isinstance(mech, Channel) or not mech.enabled:
                continue
            g += mech.conductance(state)
        return g
