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
'''Neighbor lists for pairwise (non-bonding) interactions

   Yaff works with half neighbor lists with relative vector information and with
   support for Verlet skin.

   Yaff supports only one neighbor list, which is used to evaluate all
   non-bonding interactions. The neighbor list is used by the ``ForcePartPair``
   objects. Each ``ForcePartPair`` object may have a different cutoff, of which
   the largest one determines the cutoff of the neighbor list. Unlike several
   other codes, Yaff uses one long neighbor list that contains all relevant atom
   pairs.

   The ``NeighborList`` object contains algorithms to detect whether a full rebuild
   of the neighbor list is required, or whether a recomputation of the distances
   and relative vectors is sufficient.
'''


from __future__ import division

import numpy as np

from yaff.pes.ext import neigh_dtype, nlist_status_init,\
        nlist_status_finish, nlist_build, nlist_recompute


__all__ = ['NeighborList','BondedNeighborList']


class NeighborList(object):
    '''Algorithms to keep track of all pair distances below a given rcut
    '''
    def __init__(self, system, skin=0, nlow=0, nhigh=-1, log=None,timer=None):
        """
           **Arguments:**

           system
                A System instance.

           **Optional arguments:**

           skin
                A margin added to the rcut parameter. Only when atoms are
                displaced by half this distance, the neighbor list is rebuilt
                from scratch. In the other case, the distances of the known
                pairs are just recomputed. If set to zero, the default, the
                neighbor list is rebuilt at each update.

                A reasonable skin setting can drastically improve the
                performance of the neighbor list updates. For example, when
                ``rcut`` is ``10*angstrom``, a ``skin`` of ``2*angstrom`` is
                reasonable. If the skin is set too large, the updates will
                become very inefficient. Some tuning of ``rcut`` and ``skin``
                may be beneficial.

            nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.

            nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion.
                If nlow=nhigh, the system is divided into two parts and only
                pairs involving one atom of each part will be included. This is
                useful to calculate interaction energies in Monte Carlo
                simulations
           log 
                A Screenlog object can be passed locally
                if None, the log of the system is used
                
           timer
                A TimerGroup object can be passed locally
                if None, the log of the system is used
        """
        if skin < 0:
            raise ValueError('The skin parameter must be positive.')
        if nhigh == -1:
            nhigh = system.natom
        self.system = system
        self.skin = skin
        self.rcut = 0.0
        # the neighborlist:
        self.neighs = np.empty(10, dtype=neigh_dtype)
        self.nneigh = 0
        self.rmax = None
        if nlow < 0:
            raise ValueError('nlow must be a positive number, received %d.'%nlow)
        self.nlow = nlow
        if nhigh < self.nlow:
            raise ValueError('nhigh must not be smaller than nlow, received %d.'%nhigh)
        self.nhigh = nhigh
        # for skin algorithm:
        self._pos_old = None
        self.rebuild_next = False
        if log is None:
            log=system.log
        self.log=log
        if timer is None:
            timer=system.timer
        self.timer=timer
    def request_rcut(self, rcut):
        """Make sure the internal rcut parameter is at least is high as rcut."""
        self.rcut = max(self.rcut, rcut)
        self.update_rmax()

    def update_rmax(self):
        """Recompute the ``rmax`` attribute.

           ``rmax`` determines the number of periodic images that are
           considered. when building the neighbor list. Along the a direction,
           images are taken from ``-rmax[0]`` to ``rmax[0]`` (inclusive). The
           range of images along the b and c direction are controlled by
           ``rmax[1]`` and ``rmax[2]``, respectively.

           Updating ``rmax`` may be necessary for two reasons: (i) the cutoff
           has changed, and (ii) the cell vectors have changed.
        """
        # determine the number of periodic images
        self.rmax = np.ceil((self.rcut+self.skin)/self.system.cell.rspacings-0.5).astype(int)
        if self.log.do_high:
            if len(self.rmax) == 1:
                self.log('rmax a       = %i' % tuple(self.rmax))
            elif len(self.rmax) == 2:
                self.log('rmax a,b     = %i,%i' % tuple(self.rmax))
            elif len(self.rmax) == 3:
                self.log('rmax a,b,c   = %i,%i,%i' % tuple(self.rmax))
        # Request a rebuild of the neighborlist because there is no simple way
        # to figure out whether an update is sufficient.
        self.rebuild_next = True

    def update(self):
        '''Rebuild or recompute the neighbor lists

           Based on the changes of the atomic positions or due to calls to
           ``update_rcut`` and ``update_rmax``, the neighbor lists will be
           rebuilt from scratch.

           The heavy computational work is done in low-level C routines. The
           neighbor lists array is reallocated if needed. The memory allocation
           is done in Python for convenience.
        '''
        with self.log.section('NLIST'), self.timer.section('Nlists'):
            assert self.rcut > 0

            if self._need_rebuild():
                # *rebuild* the entire neighborlist
                if self.system.cell.volume != 0:
                    if self.system.natom/self.system.cell.volume > 10:
                        raise ValueError('Atom density too high')
                # 1) make an initial status object for the neighbor list algorithm
                status = nlist_status_init(self.rmax)
                # The atom index of the first atom in pair is always at least
                # nlow. The following status initialization avoids searching
                # for excluded atom pairs in the neighbourlist build
                status[3] = self.nlow
                # 2) a loop of consecutive update/allocate calls
                last_start = 0
                while True:
                    done = nlist_build(
                        self.system.pos, self.rcut + self.skin, self.rmax,
                        self.system.cell, status, self.neighs[last_start:], self.nlow, self.nhigh
                    )
                    if done:
                        break
                    last_start = len(self.neighs)
                    new_neighs = np.empty((len(self.neighs)*3)//2, dtype=neigh_dtype)
                    new_neighs[:last_start] = self.neighs
                    self.neighs = new_neighs
                    del new_neighs
                # 3) get the number of neighbors in the list.
                self.nneigh = nlist_status_finish(status)
                if self.log.do_debug:
                    self.log('Rebuilt, size = %i' % self.nneigh)
                # 4) store the current state to check in future calls if we
                #    need to do a rebuild or a recompute.
                self._checkpoint()
                self.rebuild_next = False
            else:
                # just *recompute* the deltas and the distance in the
                # neighborlist
                nlist_recompute(self.system.pos, self._pos_old, self.system.cell, self.neighs[:self.nneigh])
                if self.log.do_debug:
                    self.log('Recomputed')

    def _checkpoint(self):
        '''Internal method called after a neighborlist rebuild.'''
        if self.skin > 0:
            # Only use the skin algorithm if this parameter is larger than zero.
            if self._pos_old is None:
                self._pos_old = self.system.pos.copy()
            else:
                self._pos_old[:] = self.system.pos

    def _need_rebuild(self):
        '''Internal method that determines if a rebuild is needed.'''
        if self.skin <= 0 or self._pos_old is None or self.rebuild_next:
            return True
        else:
            # Compute an upper bound for the maximum relative displacement.
            disp = np.sqrt(((self.system.pos - self._pos_old)**2).sum(axis=1).max())
            disp *= 2*(self.rmax.max()+1)
            if self.log.do_debug:
                self.log('Maximum relative displacement %s      Skin %s' % (self.log.length(disp), self.log.length(self.skin)))
            # Compare with skin parameter
            return disp >= self.skin


    def to_dictionary(self):
        """Transform current neighbor list into a dictionary.

           This is slow. Use this method for debugging only!
        """
        dictionary = {}
        for i in range(self.nneigh):
            key = (
                self.neighs[i]['a'], self.neighs[i]['b'], self.neighs[i]['r0'],
                self.neighs[i]['r1'], self.neighs[i]['r2']
            )
            value = np.array([
                self.neighs[i]['d'], self.neighs[i]['dx'],
                self.neighs[i]['dy'], self.neighs[i]['dz']
            ])
            dictionary[key] = value
        return dictionary


    def check(self):
        """Perform a slow internal consistency test.

           Use this for debugging only. It is assumed that self.rmax is set correctly.
        """
        # 0) Some initial tests
        assert (
            (self.neighs['a'][:self.nneigh] > self.neighs['b'][:self.nneigh]) |
            (self.neighs['r0'][:self.nneigh] != 0) |
            (self.neighs['r1'][:self.nneigh] != 0) |
            (self.neighs['r2'][:self.nneigh] != 0)
        ).all()
        # A) transform the current nlist into a set
        actual = self.to_dictionary()
        # B) Define loops of cell vectors
        if self.system.cell.nvec == 3:
            def rloops():
                for r2 in range(0, self.rmax[2]+1):
                    if r2 == 0:
                        r1_start = 0
                    else:
                        r1_start = -self.rmax[1]
                    for r1 in range(r1_start, self.rmax[1]+1):
                        if r2 == 0 and r1 == 0:
                            r0_start = 0
                        else:
                            r0_start = -self.rmax[0]
                        for r0 in range(r0_start, self.rmax[0]+1):
                            yield r0, r1, r2
        elif self.system.cell.nvec == 2:
            def rloops():
                for r1 in range(0, self.rmax[1]+1):
                    if r1 == 0:
                        r0_start = 0
                    else:
                        r0_start = -self.rmax[0]
                    for r0 in range(r0_start, self.rmax[0]+1):
                        yield r0, r1, 0

        elif self.system.cell.nvec == 1:
            def rloops():
                for r0 in range(0, self.rmax[0]+1):
                    yield r0, 0, 0
        else:
            def rloops():
                yield 0, 0, 0

        # C) Compute the nlists the slow way
        validation = {}
        nvec = self.system.cell.nvec
        for r0, r1, r2 in rloops():
            for a in range(self.system.natom):
                for b in range(a+1):
                    if r0!=0 or r1!=0 or r2!=0:
                        signs = [1, -1]
                    elif a > b:
                        signs = [1]
                    else:
                        continue
                    for sign in signs:
                        delta = self.system.pos[b] - self.system.pos[a]
                        self.system.cell.mic(delta)
                        delta *= sign
                        if nvec > 0:
                            self.system.cell.add_vec(delta, np.array([r0, r1, r2])[:nvec])
                        d = np.linalg.norm(delta)
                        if d < self.rcut + self.skin:
                            if sign == 1:
                                key = a, b, r0, r1, r2
                            else:
                                key = b, a, r0, r1, r2
                            value = np.array([d, delta[0], delta[1], delta[2]])
                            validation[key] = value

        # D) Compare
        wrong = False
        with self.log.section('NLIST'):
            for key0, value0 in validation.items():
                value1 = actual.pop(key0, None)
                if value1 is None:
                    self.log('Missing:  ', key0)
                    self.log('  Validation %s %s %s %s' % (
                        self.log.length(value0[0]), self.log.length(value0[1]),
                        self.log.length(value0[2]), self.log.length(value0[3])
                    ))
                    wrong = True
                elif abs(value0 - value1).max() > 1e-10*self.log.length.conversion:
                    self.log('Different:', key0)
                    self.log('  Actual     %s %s %s %s' % (
                        self.log.length(value1[0]), self.log.length(value1[1]),
                        self.log.length(value1[2]), self.log.length(value1[3])
                    ))
                    self.log('  Validation %s %s %s %s' % (
                        self.log.length(value0[0]), self.log.length(value0[1]),
                        self.log.length(value0[2]), self.log.length(value0[3])
                    ))
                    self.log('  Difference %10.3e %10.3e %10.3e %10.3e' %
                        tuple((value0 - value1)/self.log.length.conversion)
                    )
                    self.log('  AbsMaxDiff %10.3e' %
                        (abs(value0 - value1).max()/self.log.length.conversion)
                    )
                    wrong = True
            for key1, value1 in actual.items():
                self.log('Redundant:', key1)
                self.log('  Actual     %s %s %s %s' % (
                    self.log.length(value1[0]), self.log.length(value1[1]),
                    self.log.length(value1[2]), self.log.length(value1[3])
                ))
                wrong = True
        assert not wrong


class BondedNeighborList(NeighborList):
    '''A neighbor list that is intended for near-neighbor interactions. The
       pairs in the list are never updated, only distances are recomputed.
    '''
    def __init__(self, system, selected=[], add12=True, add13=True, add14=True,
                    add15=False,log=None,timer=None):
        '''
           **Arguments:**

           system
                A System instance.

           **Optional arguments:**

           selected
                A list containing all pairs of atoms that should be considered.
                Default: All 1-2, 1-3 and 1-4 pairs included
           log 
                A Screenlog object can be passed locally
                if None, the log of the system is used
                
           timer
                A TimerGroup object can be passed locally
                if None, the timer of the system is used
        '''
        self.system = system
        for i0 in range(system.natom):
            for i1 in system.neighs1[i0]:
                if i0 > i1 and add12: selected.append([i0,i1])
            for i2 in system.neighs2[i0]:
                if i0 > i2 and add13: selected.append([i0,i2])
            for i3 in system.neighs3[i0]:
                if i0 > i3 and add14: selected.append([i0,i3])
            for i4 in system.neighs4[i0]:
                if i0 > i4 and add15: selected.append([i0,i4])
        # Only retain unique pairs
        pairs = np.array([np.array(x) for x in set(tuple(x) for x in selected)])
        self.nneigh = pairs.shape[0]
        neighs = np.empty((self.nneigh),dtype=neigh_dtype)
        for ibond, (a,b) in enumerate(pairs):
            neighs[ibond]['a'] = a
            neighs[ibond]['b'] = b
            neighs[ibond]['r0'] = 0
            neighs[ibond]['r1'] = 0
            neighs[ibond]['r2'] = 0
            neighs[ibond]['d'] = 0.0
            neighs[ibond]['dx'] = 0.0
            neighs[ibond]['dy'] = 0.0
            neighs[ibond]['dz'] = 0.0
        self.neighs = np.sort(neighs, order=['a','b']).copy()
        del neighs, selected, pairs
        self._pos_old = system.pos.copy()
        if log is None:
            log=system.log
        self.log=log
        if timer is None:
            timer=system.timer
        self.timer=timer

    def request_rcut(self, rcut):
        # Nothing to do...
        pass

    def update_rmax(self):
        # Nothing to do...
        pass

    def update(self):
        # Simply recompute distances, no need to rebuild
        nlist_recompute(self.system.pos, self._pos_old, self.system.cell, self.neighs[:self.nneigh])
        self._pos_old[:] = self.system.pos
