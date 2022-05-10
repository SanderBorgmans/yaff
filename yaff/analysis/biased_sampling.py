# -*- coding: utf-8 -*-
# YAFF is yet another force-field code.
# Copyright (C) 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>,
# Louis Vanduyfhuys <Louis.Vanduyfhuys@UGent.be>, Center for Molecular Modeling
# (CMM), Ghent University, Ghent, Belgium; all rights reserved unless otherwise
# stated.
#
# This file is part of YAFF.
#
# YAFF is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# YAFF is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
'''Process biased sampling methods'''


from __future__ import division

import numpy as np
import h5py as h5



__all__ = ['SumHills']


class SumHills(object):
    def __init__(self, grid,log=None):
        """
           Computes a free energy profile by summing hills deposited during
           a metadyanmics simulation.

           **Argument:**

           grid
                A [N, n] NumPy array, where n is the number of collective
                variables and N is the number of grid points
           **Optional Arguments:**
           log 
                A Screenlog object can be passed locally
                if None, the global log is used
        """
        if grid.ndim==1:
            grid = np.asarray([grid]).T
        self.grid = grid
        self.ncv = self.grid.shape[1]
        self.q0s = None
        if log is None:
            from yaff.log import log
        self.log=log

    def compute_fes(self):
        if self.q0s is None:
            raise ValueError("Hills not initialized")
        ngauss = self.q0s.shape[0]
        if self.tempering != 0.0:
            prefactor = self.tempering/(self.tempering+self.T)
        else: prefactor = 1.0
        # Compute exponential argument
        deltas = np.diagonal(np.subtract.outer(self.grid, self.q0s), axis1=1, axis2=3).copy()
        # Apply minimum image convention
        if self.periodicities is not None:
            for icv in range(self.ncv):
                if self.periodicities[icv] is None: continue
                # Translate (q-q0) over integer multiple of the period P, so it
                # ends up in [-P/2,P/2]
                deltas[:,:,icv] -= np.floor(0.5+deltas[:,:,icv]/
                    self.periodicities[icv])*self.periodicities[icv]
        exparg = deltas*deltas
        exparg = np.multiply(exparg, 0.5/self.sigmas**2)
        exparg = np.sum(exparg, axis=2)
        exponents = np.exp(-exparg)
        # Compute the bias energy
        fes = -prefactor*np.sum(np.multiply(exponents, self.Ks),axis=1)
        return fes

    def set_hills(self, q0s, Ks, sigmas, tempering=0.0, T=None, periodicities=None):
        # Safety checks
        assert q0s.shape[1]==self.ncv
        assert sigmas.shape[0]==self.ncv
        assert q0s.shape[0]==Ks.shape[0]
        if tempering != 0.0 and T is None:
            raise ValueError("For a well-tempered MTD run, the temperature "
                "has to be specified")
        self.q0s = q0s
        self.sigmas = sigmas
        self.Ks = Ks
        self.tempering = tempering
        self.T = T
        self.periodicities = periodicities
        if self.log.do_medium:
            with self.log.section("SUMHILL"):
                self.log("Found %d collective variables and %d Gaussian hills"%(self.ncv,self.q0s.shape[0]))

    def load_hdf5(self, fn, T=None):
        """
           Read information from HDF5 file

           **Arguments:**

           fn
                A HDF5 filename containing a hills group. If this concerns a well-
                tempered MTD run, the simulation temperature should be provided
                Otherwise, it will be read from the HDF5 file.
        """
        with h5.File(fn,'r') as f:
            q0s = f['hills/q0'][:]
            Ks = f['hills/K'][:]
            sigmas = f['hills/sigma'][:]
            tempering = f['hills'].attrs['tempering']
            if tempering!=0.0:
                if T is None:
                    if not 'trajectory/temp' in f:
                        raise ValueError("For a well-tempered MTD run, the temperature "
                            "should be specified or readable from the trajectory/temp "
                            "group in the HDF5 file")
                    T = np.mean(f['trajectory/temp'][:])
                if self.log.do_medium:
                    self.log("Well-tempered MTD run: T = %s deltaT = %s"%(self.log.temperature(T), self.log.temperature(tempering)))
            if 'hills/periodicities' in f:
                periodicities = f['hills/periodicities'][:]
            else:
                periodicities = None
            self.set_hills(q0s, Ks, sigmas, tempering=tempering, T=T, periodicities=periodicities)
