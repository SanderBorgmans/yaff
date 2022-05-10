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
'''Force field models

   This module contains the force field computation interface that is used by
   the :mod:`yaff.sampling` package.

   The ``ForceField`` class is the main item in this module. It acts as
   container for instances of subclasses of ``ForcePart``. Each ``ForcePart``
   subclass implements a typical contribution to the force field energy, e.g.
   ``ForcePartValence`` computes covalent interactions, ``ForcePartPair``
   computes pairwise (non-bonding) interactions, and so on. The ``ForceField``
   object also contains one neighborlist object, which is used by all
   ``ForcePartPair`` objects. Actual computations are done through the
   ``compute`` method of the ``ForceField`` object, which calls the ``compute``
   method of all the ``ForceParts`` and adds up the results.
'''


from __future__ import division

import sys
import numpy as np

from yaff.pes.ext import compute_ewald_reci, compute_ewald_reci_dd, \
    compute_ewald_corr, compute_ewald_corr_dd, compute_ewald_prefactors, \
    compute_ewald_structurefactors, compute_ewald_deltae, PairPotEI, \
    PairPotLJ, PairPotMM3, PairPotMM3CAP, PairPotGrimme, compute_grid3d
from yaff.pes.dlist import DeltaList
from yaff.pes.iclist import InternalCoordinateList
from yaff.pes.vlist import ValenceList, ValenceTerm
from yaff.pes.bias import BiasPotential
from yaff.system import System

__all__ = [
    'ForcePart', 'ForceField', 'ForcePartPair', 'ForcePartEwaldReciprocal',
    'ForcePartEwaldReciprocalDD', 'ForcePartEwaldCorrectionDD',
    'ForcePartEwaldCorrection', 'ForcePartEwaldNeutralizing',
    'ForcePartValence', 'ForcePartBias', 'ForcePartPressure', 'ForcePartGrid',
    'ForcePartTailCorrection', 'ForcePartEwaldReciprocalInteraction', 'ForcePartTIP4P','ForcePartQTIP4P'
]


class ForcePart(object):
    '''Base class for anything that can compute energies (and optionally gradient
       and virial) for a ``System`` object.
    '''
    def __init__(self, name, system,log=None, timer=None):
        """
           **Arguments:**

           name
                A name for this part of the force field. This name must adhere
                to the following conventions: all lower case, no white space,
                and short. It is used to construct part_* attributes in the
                ForceField class, where * is the name.

           system
                The system to which this part of the FF applies.
          **Optional Arguments**
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        """
        self.name = name
        # backup copies of last call to compute:
        self.energy = 0.0
        self.gpos = np.zeros((system.natom, 3), float)
        self.vtens = np.zeros((3, 3), float)
        self.clear()
        if log is None:
            from yaff.log import log
        self.log=log
        if timer is None:
            from yaff.log import timer
        self.timer=timer

    def clear(self):
        """Fill in nan values in the cached results to indicate that they have
           become invalid.
        """
        self.energy = np.nan
        self.gpos[:] = np.nan
        self.vtens[:] = np.nan

    def update_rvecs(self, rvecs):
        '''Let the ``ForcePart`` object know that the cell vectors have changed.

           **Arguments:**

           rvecs
                The new cell vectors.
        '''
        self.clear()

    def update_pos(self, pos):
        '''Let the ``ForcePart`` object know that the atomic positions have changed.

           **Arguments:**

           pos
                The new atomic coordinates.
        '''
        self.clear()

    def compute(self, gpos=None, vtens=None):
        """Compute the energy and optionally some derivatives for this FF (part)

           The only variable inputs for the compute routine are the atomic
           positions and the cell vectors, which can be changed through the
           ``update_rvecs`` and ``update_pos`` methods. All other aspects of
           a force field are considered to be fixed between subsequent compute
           calls. If changes other than positions or cell vectors are needed,
           one must construct new ``ForceField`` and/or ``ForcePart`` objects.

           **Optional arguments:**

           gpos
                The derivatives of the energy towards the Cartesian coordinates
                of the atoms. ('g' stands for gradient and 'pos' for positions.)
                This must be a writeable numpy array with shape (N, 3) where N
                is the number of atoms.

           vtens
                The force contribution to the pressure tensor. This is also
                known as the virial tensor. It represents the derivative of the
                energy towards uniform deformations, including changes in the
                shape of the unit cell. (v stands for virial and 'tens' stands
                for tensor.) This must be a writeable numpy array with shape (3,
                3). Note that the factor 1/V is not included.

           The energy is returned. The optional arguments are Fortran-style
           output arguments. When they are present, the corresponding results
           are computed and **added** to the current contents of the array.
        """
        if gpos is None:
            my_gpos = None
        else:
            my_gpos = self.gpos
            my_gpos[:] = 0.0
        if vtens is None:
            my_vtens = None
        else:
            my_vtens = self.vtens
            my_vtens[:] = 0.0
        self.energy = self._internal_compute(my_gpos, my_vtens)
        if np.isnan(self.energy):
            raise ValueError('The energy is not-a-number (nan).')
        if gpos is not None:
            if np.isnan(my_gpos).any():
                raise ValueError('Some gpos element(s) is/are not-a-number (nan).')
            gpos += my_gpos
        if vtens is not None:
            if np.isnan(my_vtens).any():
                raise ValueError('Some vtens element(s) is/are not-a-number (nan).')
            vtens += my_vtens
        return self.energy

    def _internal_compute(self, gpos, vtens):
        '''Subclasses implement their compute code here.'''
        raise NotImplementedError


class ForceField(ForcePart):
    '''A complete force field model.'''
    def __init__(self, system, parts, nlist=None,log=None, timer=None):
        """
           **Arguments:**

           system
                An instance of the ``System`` class.

           parts
                A list of instances of sublcasses of ``ForcePart``. These are
                the different types of contributions to the force field, e.g.
                valence interactions, real-space electrostatics, and so on.

           **Optional arguments:**

           nlist
                A ``NeighborList`` instance. This is required if some items in the
                parts list use this nlist object.
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        """
        ForcePart.__init__(self, 'all', system,log=log, timer=timer)
        self.system = system
        self.parts = []
        self.nlist = nlist
        self.needs_nlist_update = nlist is not None
        for part in parts:
            self.add_part(part)
        if self.log.do_medium:
            with self.log.section('FFINIT'):
                self.log('Force field with %i parts:&%s.' % (
                    len(self.parts), ', '.join(part.name for part in self.parts)
                ))
                self.log('Neighborlist present: %s' % (self.nlist is not None))

    def add_part(self, part):
        self.parts.append(part)
        # Make the parts also accessible as simple attributes.
        name = 'part_%s' % part.name
        if name in self.__dict__:
            raise ValueError('The part %s occurs twice in the force field.' % name)
        self.__dict__[name] = part

    @classmethod
    def generate(cls, system, parameters,log=None, timer=None, **kwargs):
        """Create a force field for the given system with the given parameters.

           **Arguments:**

           system
                An instance of the System class

           parameters
                Three types are accepted: (i) the filename of the parameter
                file, which is a text file that adheres to YAFF parameter
                format, (ii) a list of such filenames, or (iii) an instance of
                the Parameters class.

           See the constructor of the :class:`yaff.pes.generator.FFArgs` class
           for the available optional arguments.

           This method takes care of setting up the FF object, and configuring
           all the necessary FF parts. This is a lot easier than creating an FF
           with the default constructor. Parameters for atom types that are not
           present in the system, are simply ignored.
        """
        if log is None:
            from yaff.log import log
        if timer is None:
            from yaff.log import timer
        if system.ffatype_ids is None:
            raise ValueError('The generators needs ffatype_ids in the system object.')
        with log.section('GEN'), timer.section('Generator'):
            from yaff.pes.generator import apply_generators, FFArgs
            from yaff.pes.parameters import Parameters
            if log.do_medium:
                log('Generating force field from %s' % str(parameters))
            if not isinstance(parameters, Parameters):
                parameters = Parameters.from_file(parameters)
            ff_args = FFArgs(log=log,timer=timer,**kwargs)
            apply_generators(system, parameters, ff_args)
            return ForceField(system, ff_args.parts, ff_args.nlist,log=log,timer=timer)

    def update_rvecs(self, rvecs):
        '''See :meth:`yaff.pes.ff.ForcePart.update_rvecs`'''
        ForcePart.update_rvecs(self, rvecs)
        self.system.cell.update_rvecs(rvecs)
        if self.nlist is not None:
            self.nlist.update_rmax()
            self.needs_nlist_update = True

    def update_pos(self, pos):
        '''See :meth:`yaff.pes.ff.ForcePart.update_pos`'''
        ForcePart.update_pos(self, pos)
        self.system.pos[:] = pos
        if self.nlist is not None:
            self.needs_nlist_update = True

    def _internal_compute(self, gpos, vtens):
        if self.needs_nlist_update:
            self.nlist.update()
            self.needs_nlist_update = False
        result = sum([part.compute(gpos, vtens) for part in self.parts])
        return result


class ForcePartPair(ForcePart):
    '''A pairwise (short-range) non-bonding interaction term.

       This part can be used for the short-range electrostatics, Van der Waals
       terms, etc. Currently, one has to use multiple ``ForcePartPair``
       objects in a ``ForceField`` in order to combine different types of pairwise
       energy terms, e.g. to combine an electrostatic term with a Van der
       Waals term. (This may be changed in future to improve the computational
       efficiency.)
    '''
    def __init__(self, system, nlist, scalings, pair_pot, log=None, timer=None):
        '''
           **Arguments:**

           system
                The system to which this pairwise interaction applies.

           nlist
                A ``NeighborList`` object. This has to be the same as the one
                passed to the ForceField object that contains this part.

           scalings
                A ``Scalings`` object. This object contains all the information
                about the energy scaling of pairwise contributions that are
                involved in covalent interactions. See
                :class:`yaff.pes.scalings.Scalings` for more details.

           pair_pot
                An instance of the ``PairPot`` built-in class from
                :mod:`yaff.pes.ext`.
           **Optional Arguments**
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        '''
        ForcePart.__init__(self, 'pair_%s' % pair_pot.name, system, log=log, timer=timer)
        self.nlist = nlist
        self.scalings = scalings
        self.pair_pot = pair_pot
        self.nlist.request_rcut(pair_pot.rcut)
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()
                self.log('  scalings:          %5.3f %5.3f %5.3f' % (scalings.scale1, scalings.scale2, scalings.scale3))
                self.log('  real space cutoff: %s' % self.log.length(pair_pot.rcut))
                tr = pair_pot.get_truncation()
                if tr is None:
                    self.log('  truncation:     none')
                else:
                    try:
                        self.log('  truncation:     %s' % tr.get_log(self.log))
                    except TypeError:
                        print(tr, type(tr))
                        sys.exit(0)
                self.pair_pot.get_log(self.log)
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('PP %s' % self.pair_pot.name):
            return self.pair_pot.compute(self.nlist.neighs, self.scalings.stab, gpos, vtens, self.nlist.nneigh)


class ForcePartEwaldReciprocal(ForcePart):
    '''The long-range contribution to the electrostatic interaction in 3D
       periodic systems.
    '''
    def __init__(self, system, alpha, gcut=0.35, dielectric=1.0, nlow=0, nhigh=-1,log=None, timer=None):
        '''
           **Arguments:**

           system
                The system to which this interaction applies.

           alpha
                The alpha parameter in the Ewald summation method.

           **Optional arguments:**

           gcut
                The cutoff in reciprocal space.

           dielectric
                The scalar relative permittivity of the system.

           nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.

           nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion.
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        '''
        ForcePart.__init__(self, 'ewald_reci', system,log=log, timer=timer)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell.')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.gcut = gcut
        self.dielectric = dielectric
        self.update_gmax()
        self.work = np.empty(system.natom*2)
        self.nlow, self.nhigh = check_nlow_nhigh(system, nlow, nhigh)
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()
                self.log('  alpha:                 %s' % self.log.invlength(self.alpha))
                self.log('  gcut:                  %s' % self.log.invlength(self.gcut))
                self.log('  relative permittivity: %5.3f' % self.dielectric)
                self.log.hline()


    def update_gmax(self):
        '''This routine must be called after the attribute self.gmax is modified.'''
        self.gmax = np.ceil(self.gcut/self.system.cell.gspacings-0.5).astype(int)
        if self.log.do_debug:
            with self.log.section('EWALD'):
                self.log('gmax a,b,c   = %i,%i,%i' % tuple(self.gmax))

    def update_rvecs(self, rvecs):
        '''See :meth:`yaff.pes.ff.ForcePart.update_rvecs`'''
        ForcePart.update_rvecs(self, rvecs)
        self.update_gmax()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Ewald reci.'):
            return compute_ewald_reci(
                self.system.pos, self.system.charges, self.system.cell, self.alpha,
                self.gmax, self.gcut, self.dielectric, gpos, self.work, vtens, self.nlow, self.nhigh
            )


class ForcePartEwaldReciprocalDD(ForcePart):
    '''The long-range contribution to the dipole-dipole
       electrostatic interaction in 3D periodic systems.
    '''
    def __init__(self, system, alpha, gcut=0.35, nlow=0, nhigh=-1,log=None, timer=None):
        '''
           **Arguments:**

           system
                The system to which this interaction applies.

           alpha
                The alpha parameter in the Ewald summation method.

           gcut
                The cutoff in reciprocal space.

           nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.

           nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion.
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        '''
        ForcePart.__init__(self, 'ewald_reci', system,log=log, timer=timer)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell.')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        if system.dipoles is None:
            raise ValueError('The system does not have dipoles.')
        self.system = system
        self.alpha = alpha
        self.gcut = gcut
        self.update_gmax()
        self.work = np.empty(system.natom*2)
        self.nlow, self.nhigh = check_nlow_nhigh(system, nlow, nhigh)
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()
                self.log('  alpha:             %s' % self.log.invlength(self.alpha))
                self.log('  gcut:              %s' % self.log.invlength(self.gcut))
                self.log.hline()


    def update_gmax(self):
        '''This routine must be called after the attribute self.gmax is modified.'''
        self.gmax = np.ceil(self.gcut/self.system.cell.gspacings-0.5).astype(int)
        if self.log.do_debug:
            with self.log.section('EWALD'):
                self.log('gmax a,b,c   = %i,%i,%i' % tuple(self.gmax))

    def update_rvecs(self, rvecs):
        '''See :meth:`yaff.pes.ff.ForcePart.update_rvecs`'''
        ForcePart.update_rvecs(self, rvecs)
        self.update_gmax()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Ewald reci.'):
            return compute_ewald_reci_dd(
                self.system.pos, self.system.charges, self.system.dipoles, self.system.cell, self.alpha,
                self.gmax, self.gcut, gpos, self.work, vtens, self.nlow, self.nhigh
            )


class ForcePartEwaldCorrection(ForcePart):
    '''Correction for the double counting in the long-range term of the Ewald sum.

       This correction is only needed if scaling rules apply to the short-range
       electrostatics.
    '''
    def __init__(self, system, alpha, scalings, dielectric=1.0, nlow=0, nhigh=-1,log=None, timer=None):
        '''
           **Arguments:**

           system
                The system to which this interaction applies.

           alpha
                The alpha parameter in the Ewald summation method.

           scalings
                A ``Scalings`` object. This object contains all the information
                about the energy scaling of pairwise contributions that are
                involved in covalent interactions. See
                :class:`yaff.pes.scalings.Scalings` for more details.

           **Optional arguments:**

           dielectric
                The scalar relative permittivity of the system.

           nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.

           nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion.
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        '''
        ForcePart.__init__(self, 'ewald_cor', system,log=log, timer=timer)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.dielectric = dielectric
        self.nlow, self.nhigh = check_nlow_nhigh(system, nlow, nhigh)
        self.scalings = scalings
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()
                self.log('  alpha:             %s' % self.log.invlength(self.alpha))
                self.log('  relative permittivity   %5.3f' % self.dielectric)
                self.log('  scalings:          %5.3f %5.3f %5.3f' % (scalings.scale1, scalings.scale2, scalings.scale3))
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Ewald corr.'):
            return compute_ewald_corr(
                self.system.pos, self.system.charges, self.system.cell,
                self.alpha, self.scalings.stab, self.dielectric, gpos, vtens, self.nlow, self.nhigh
            )


class ForcePartEwaldCorrectionDD(ForcePart):
    '''Correction for the double counting in the long-range term of the Ewald sum.

       This correction is only needed if scaling rules apply to the short-range
       electrostatics.
    '''
    def __init__(self, system, alpha, scalings, nlow=0, nhigh=-1,log=None, timer=None):
        '''
           **Arguments:**

           system
                The system to which this interaction applies.

           alpha
                The alpha parameter in the Ewald summation method.

           scalings
                A ``Scalings`` object. This object contains all the information
                about the energy scaling of pairwise contributions that are
                involved in covalent interactions. See
                :class:`yaff.pes.scalings.Scalings` for more details.

           nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.

           nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion.
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        '''
        ForcePart.__init__(self, 'ewald_cor', system,log=log, timer=timer)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.nlow, self.nhigh = check_nlow_nhigh(system, nlow, nhigh)
        self.scalings = scalings
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()
                self.log('  alpha:             %s' % self.log.invlength(self.alpha))
                self.log('  scalings:          %5.3f %5.3f %5.3f' % (scalings.scale1, scalings.scale2, scalings.scale3))
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Ewald corr.'):
            return compute_ewald_corr_dd(
                self.system.pos, self.system.charges, self.system.dipoles, self.system.cell,
                self.alpha, self.scalings.stab, gpos, vtens, self.nlow, self.nhigh
            )

class ForcePartEwaldNeutralizing(ForcePart):
    '''Neutralizing background correction for 3D periodic systems that are
       charged.

       This term is only required of the system is not neutral.
    '''
    def __init__(self, system, alpha, dielectric=1.0, nlow=0, nhigh=-1,
                fluctuating_charges=False,log=None, timer=None):
        '''
           **Arguments:**

           system
                The system to which this interaction applies.

           alpha
                The alpha parameter in the Ewald summation method.

           **Optional arguments:**

           dielectric
                The scalar relative permittivity of the system.

           nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.

           nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion.

           fluctuating_charges
                Boolean indicating whether charges (and radii) are allowed to
                change during a simulation. If set to False, some factors can
                be precomputed at the start of the simulation.
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        '''
        ForcePart.__init__(self, 'ewald_neut', system,log=log, timer=timer)
        if not system.cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell')
        if system.charges is None:
            raise ValueError('The system does not have charges.')
        self.system = system
        self.alpha = alpha
        self.dielectric = dielectric
        self.nlow, self.nhigh = check_nlow_nhigh(system, nlow, nhigh)
        self.fluctuating_charges = fluctuating_charges
        if not self.fluctuating_charges:
            fac = self.system.charges[:].sum()**2/self.alpha**2
            fac -= self.system.charges[:self.nlow].sum()**2/self.alpha**2
            fac -= self.system.charges[self.nhigh:].sum()**2/self.alpha**2
            if self.system.radii is not None:
                fac -= self.system.charges.sum()*np.sum( self.system.charges*self.system.radii**2 )
                fac += self.system.charges[:self.nlow].sum()*np.sum( self.system.charges[:self.nlow]*self.system.radii[:self.nlow]**2)
                fac += self.system.charges[self.nhigh:].sum()*np.sum( self.system.charges[self.nhigh:]*self.system.radii[self.nhigh:]**2)
            self.prefactor = fac*np.pi/(2.0*self.dielectric)
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()
                self.log('  alpha:                   %s' % self.log.invlength(self.alpha))
                self.log('  relative permittivity:   %5.3f' % self.dielectric)
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Ewald neut.'):
            if not self.fluctuating_charges:
                fac = self.prefactor/self.system.cell.volume
            else:
                #TODO: interaction of dipoles with background? I think this is zero, need proof...
                fac = self.system.charges[:].sum()**2/self.alpha**2
                fac -= self.system.charges[:self.nlow].sum()**2/self.alpha**2
                fac -= self.system.charges[self.nhigh:].sum()**2/self.alpha**2
                if self.system.radii is not None:
                    fac -= self.system.charges.sum()*np.sum( self.system.charges*self.system.radii**2 )
                    fac += self.system.charges[:self.nlow].sum()*np.sum( self.system.charges[:self.nlow]*self.system.radii[:self.nlow]**2)
                    fac += self.system.charges[self.nhigh:].sum()*np.sum( self.system.charges[self.nhigh:]*self.system.radii[self.nhigh:]**2)
                fac *= np.pi/(2.0*self.system.cell.volume*self.dielectric)
            if vtens is not None:
                vtens.ravel()[::4] -= fac
        return fac


class ForcePartValence(ForcePart):
    '''The covalent part of a force-field model.

       The covalent force field is implemented in a three-layer approach,
       similar to the implementation of a neural network:

       (0. Optional, not used by default. A layer that computes centers of mass for groups
           of atoms.)

       1. The first layer consists of a :class:`yaff.pes.dlist.DeltaList` object
          that computes all the relative vectors needed for the internal
          coordinates in the covalent energy terms. This list is automatically
          built up as energy terms are added with the ``add_term`` method. This
          list also takes care of transforming `derivatives of the energy
          towards relative vectors` into `derivatives of the energy towards
          Cartesian coordinates and the virial tensor`.

       2. The second layer consist of a
          :class:`yaff.pes.iclist.InternalCoordinateList` object that computes
          the internal coordinates, based on the ``DeltaList``. This list is
          also automatically built up as energy terms are added. The same list
          is also responsible for transforming `derivatives of the energy
          towards internal coordinates` into `derivatives of the energy towards
          relative vectors`.

       3. The third layers consists of a :class:`yaff.pes.vlist.ValenceList`
          object. This list computes the covalent energy terms, based on the
          result in the ``InternalCoordinateList``. This list also computes the
          derivatives of the energy terms towards the internal coordinates.

       The computation of the covalent energy is the so-called `forward code
       path`, which consists of running through steps 1, 2 and 3, in that order.
       The derivatives of the energy are computed in the so-called `backward
       code path`, which consists of taking steps 1, 2 and 3 in reverse order.
       This basic idea of back-propagation for the computation of derivatives
       comes from the field of neural networks. More details can be found in the
       chapter, :ref:`dg_sec_backprop`.
    '''
    def __init__(self, system, comlist=None,log=None, timer=None):
        '''
           Parameters
           ----------

           system
                An instance of the ``System`` class.
           comlist
                An optional layer to derive centers of mass from the atomic positions.
                These centers of mass are used as input for the first layer, the relative
                vectors.
           **Optional Parameters:**

           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        '''
        ForcePart.__init__(self, 'valence', system,log=log, timer=timer)
        self.comlist = comlist
        self.dlist = DeltaList(system if comlist is None else comlist)
        self.iclist = InternalCoordinateList(self.dlist)
        self.vlist = ValenceList(self.iclist)
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()

    def add_term(self, term):
        '''Add a new term to the covalent force field.

           **Arguments:**

           term
                An instance of the class :class:`yaff.pes.ff.vlist.ValenceTerm`.

           In principle, one should add all energy terms before calling the
           ``compute`` method, but with the current implementation of Yaff,
           energy terms can be added at any time. (This may change in future.)
        '''
        if self.log.do_high:
            with self.log.section('VTERM'):
                self.log('%7i&%s %s' % (self.vlist.nv, term.get_log(), ' '.join(ic.get_log() for ic in term.ics)))
        self.vlist.add_term(term)

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Valence'):
            if self.comlist is not None:
                self.comlist.forward()
            self.dlist.forward()
            self.iclist.forward()
            energy = self.vlist.forward()
            if not ((gpos is None) and (vtens is None)):
                self.vlist.back()
                self.iclist.back()
                if self.comlist is None:
                    self.dlist.back(gpos, vtens)
                else:
                    self.comlist.gpos[:] = 0.0
                    self.dlist.back(self.comlist.gpos, vtens)
                    self.comlist.back(gpos)
            return energy


class ForcePartBias(ForcePart):
    '''Biasing potential that can be used in advanced molecular dynamics
       methods such as umbrella sampling and metadynamics.

       Terms can be added using the ``add_term`` method, where the argument is
       either an instance of ``BiasPotential`` or ``ValenceTerm``.
       In many cases, a bias term is very similar to a conventional force-field
       term, such as a harmonic bond stretch. In such a case, it is advisable
       to make use of the ``InternalCoordinate`` and ``ValenceTerm`` classes
       to construct the contribution to the biasing potential.
       If this is not possible, for instance a harmonic restraint of the cell
       volume, an instance of ``CollectiveVariable`` can be used together with
       an instance of the ``BiasPotential`` class.
    '''
    def __init__(self, system, comlist=None,log=None, timer=None):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class.

           **Optional arguments:**

           comlist
                An optional layer to derive centers of mass from the atomic positions.
                These centers of mass are used as input for the first layer, the relative
                vectors.
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        '''
        ForcePart.__init__(self, 'bias', system,log=log, timer=timer)
        self.system = system
        self.valence = ForcePartValence(system,log=log,timer=timer)
        if comlist is not None:
            self.valence_com = ForcePartValence(system, comlist=comlist,log=log,timer=timer)
        else:
            self.valence_com = None
        self.terms = []
        # The terms contributing to the bias potential are divided into three
        # categories:
        #   0) instances of BiasPotential
        #   1) instances of ValenceTerm with a regular DeltaList
        #   2) instances of ValenceTerm with a COMList
        # The following list facilitates looking up the terms after they have
        # been added
        self.term_lookup = []
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()

    def add_term(self, term, use_comlist=False):
        '''Add a new term to the bias potential.

           **Arguments:**

           term
                An instance of the class :class:`yaff.pes.ff.vlist.ValenceTerm`
                or an instance of the class
                :class:`yaff.pes.ff.vias.BiasPotential`

           **Optional arguments:**

           use_comlist
                Boolean indicating whether the comlist should be used for
                adding this ValenceTerm
        '''
        if isinstance(term, ValenceTerm):
            if use_comlist:
                if self.valence_com is None:
                    raise TypeError("No COMList was provided when setting up the ForcePartBias")
                self.term_lookup.append( (2,self.valence_com.vlist.nv) )
                # Keep track of the index this term gets in the ValenceList
                self.valence_com.vlist.add_term(term)
            else:
                self.term_lookup.append( (1,self.valence.vlist.nv) )
                # Add to the ValenceList
                self.valence.vlist.add_term(term)
            if self.log.do_high:
                with self.log.section('BIAS'):
                    self.log('%7i&%s %s' % (len(self.terms), term.get_log(), ' '.join(ic.get_log() for ic in term.ics)))
        elif isinstance(term, BiasPotential):
            self.term_lookup.append( (0,len(self.terms)))
            if self.log.do_high:
                with self.log.section('BIAS'):
                    self.log('%7i&%s %s' % (len(self.terms), term.get_log(), ' '.join(cv.get_log() for cv in term.cvs)))
        else:
            raise NotImplementedError
        self.terms.append(term)

    def get_term_energy(self, index):
        kind, iterm = self.term_lookup[index]
        if kind==0:
            return self.terms[index].compute()
        elif kind==1:
            return self.valence.vlist.vtab[iterm]['energy']
        elif kind==2:
            return self.valence_com.vlist.vtab[iterm]['energy']

    def get_term_energies(self):
        '''
        Return a NumPy array with the energies associated with all terms
        contributing to the bias potential.
        '''
        energies = np.array([self.get_term_energy(index) for index in range(len(self.terms))])
        return energies

    def get_term_cv_values(self, index):
        '''
        Return a NumPy array with values of collective variables associated
        with a certain term.

           **Arguments:**

           index
                The index of the term in question.
        '''
        kind, iterm = self.term_lookup[index]
        term = self.terms[index]
        if kind==0:
            return np.array([cv.compute() for cv in term.cvs])
        else:
            if kind==1:
                iclist = self.valence.iclist
            elif kind==2:
                iclist = self.valence_com.iclist
            cv_values = []
            # Loop over all internal coordinates for this term
            for index in term.get_ic_indexes(iclist):
                cv_values.append(iclist.ictab[index]['value'])
            return np.asarray(cv_values)

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Bias'):
            energy = 0.0
            # ValenceTerms
            energy += self.valence._internal_compute(gpos, vtens)
            if self.valence_com is not None:
                energy += self.valence_com._internal_compute(gpos, vtens)
            # BiasPotentials
            if gpos is None:
                my_gpos = None
            else:
                my_gpos = np.zeros((self.system.natom,3))
            if vtens is None:
                my_vtens = None
            else:
                my_vtens = np.zeros((3,3))
            for term in self.terms:
                if isinstance(term, ValenceTerm): continue
                energy += term.compute(gpos=my_gpos,vtens=my_vtens)
                if gpos is not None: gpos[:] += my_gpos
                if vtens is not None: vtens[:] += my_vtens
            return energy


class ForcePartPressure(ForcePart):
    '''Applies a constant istropic pressure.'''
    def __init__(self, system, pext,log=None, timer=None):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class.

           pext
                The external pressure. (Positive will shrink the system.) In
                case of 2D-PBC, this is the surface tension. In case of 1D, this
                is the linear strain.
           **Optiona Arguments**
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used

           This force part is only applicable to systems that are periodic.
        '''
        if system.cell.nvec == 0:
            raise ValueError('The system must be periodic in order to apply a pressure')
        ForcePart.__init__(self, 'press', system,log=log, timer=timer)
        self.system = system
        self.pext = pext
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Valence'):
            cell = self.system.cell
            if (vtens is not None):
                rvecs = cell.rvecs
                if cell.nvec == 1:
                    vtens += self.pext/cell.volume*np.outer(rvecs[0], rvecs[0])
                elif cell.nvec == 2:
                    vtens += self.pext/cell.volume*(
                          np.dot(rvecs[0], rvecs[0])*np.outer(rvecs[1], rvecs[1])
                        + np.dot(rvecs[0], rvecs[0])*np.outer(rvecs[1], rvecs[1])
                        - np.dot(rvecs[1], rvecs[0])*np.outer(rvecs[0], rvecs[1])
                        - np.dot(rvecs[0], rvecs[1])*np.outer(rvecs[1], rvecs[0])
                    )
                elif cell.nvec == 3:
                    gvecs = cell.gvecs
                    vtens += self.pext*cell.volume*np.identity(3)
                else:
                    raise NotImplementedError
            return cell.volume*self.pext


class ForcePartGrid(ForcePart):
    '''Energies obtained by grid interpolation.'''
    def __init__(self, system, grids,log=None, timer=None):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class.

           grids
                A dictionary with (ffatype, grid) items. Each grid must be a
                three-dimensional array with energies.
           **Optional Arguments**
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used

           This force part is only applicable to systems that are 3D periodic.
        '''
        if system.cell.nvec != 3:
            raise ValueError('The system must be 3d periodic for the grid term.')
        for grid in grids.values():
            if grid.ndim != 3:
                raise ValueError('The energy grids must be 3D numpy arrays.')
        ForcePart.__init__(self, 'grid', system,log=log, timer=timer)
        self.system = system
        self.grids = grids
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        with self.timer.section('Grid'):
            if gpos is not None:
                raise NotImplementedError('Cartesian gradients are not supported yet in ForcePartGrid')
            if vtens is not None:
                raise NotImplementedError('Cell deformation are not supported by ForcePartGrid')
            cell = self.system.cell
            result = 0
            for i in range(self.system.natom):
                grid = self.grids[self.system.get_ffatype(i)]
                result += compute_grid3d(self.system.pos[i], cell, grid)
            return result


class ForcePartTailCorrection(ForcePart):
    '''Corrections to energy and virial tensor to compensate for neglecting
    pair potentials at long range'''
    def __init__(self, system, part_pair, nlow=0, nhigh=-1,log=None, timer=None):
        '''
           **Arguments:**

           system
                An instance of the ``System`` class.

           part_pair
                An instance of the ``PairPot`` class.

           This force part is only applicable to systems that are 3D periodic.

           **Optional arguments:**

           nlow
                Atom pairs are only included if at least one atom index is
                higher than or equal to nlow. The default nlow=0 means no
                exclusion.

           nhigh
                Atom pairs are only included if at least one atom index is
                smaller than nhigh. The default nhigh=-1 means no exclusion.
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        '''
        if system.cell.nvec != 3:
            raise ValueError('Tail corrections can only be applied to 3D periodic systems')
        if part_pair.name in ['pair_ei','pair_eidip']:
            raise ValueError('Tail corrections are divergent for %s'%part_pair.name)
        super(ForcePartTailCorrection, self).__init__('tailcorr_%s'%(part_pair.name), system,log=log, timer=timer)
        self.nlow, self.nhigh = check_nlow_nhigh(system, nlow, nhigh)
        self.ecorr, self.wcorr = part_pair.pair_pot.prepare_tailcorrections(system.natom, self.nlow, self.nhigh)
        self.system = system
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('Force part: %s' % self.name)
                self.log.hline()

    def _internal_compute(self, gpos, vtens):
        if vtens is not None:
            w = 2.0*np.pi*self.wcorr/self.system.cell.volume
            vtens[0,0] += w
            vtens[1,1] += w
            vtens[2,2] += w
        return 2.0*np.pi*self.ecorr/self.system.cell.volume


class ForcePartEwaldReciprocalInteraction(ForcePart):
    r'''The reciprocal part of the Ewald summation, not the entire energy but
       only interactions between parts of the system. This allows a
       computationally very efficient evaluation of the energy difference when
       a limited number of atoms are moved, and is thus mostly useful in MC
       simulations. Although it is technically a subclass of ForcePart, it will
       not actually contribute to a ForceField. Because this class has to
       flexibility to handle varying numbers of atoms, it is only useful through
       direct calls of `compute_deltae`

       The reciprocal part of the Ewald summation for a set of N atoms is given
       by:

       .. math:: E = \frac{4\pi}{V} \sum_{\mathbf{k}} |S(\mathbf{k})|^2 \
                 \frac{e^{-\frac{k^2}{4\alpha^2}}}{k^2} \\

       where the so-called structure factors are given:

       .. math:: S(\mathbf{k}) = \sum_{i=1}^{N} q_i \
                 e^{j\mathbf{k}\cdot\mathbf{r}_i}

       Suppose that we want to compute the interaction energy with a set of
       M other atoms. This can be done as follows:

       .. math:: \Delta S(\mathbf{k}) = \sum_{i=1}^{M} q_i \
                 e^{j\mathbf{k}\cdot\mathbf{r}_i}

       Using the change in structure factors, the corresponding energy change
       is given by:

       .. math:: E = \frac{4\pi}{V} \sum_{\mathbf{k}} \left[ \bar{S}\Delta S \
                 + S\bar{\Delta S} \right] \
                 \frac{e^{-\frac{k^2}{4\alpha^2}}}{k^2} \\

       The structure factors are stored as an attribute of this Class. This
       makes it easy to add the contribution from new atoms to the existing
       structure factors, and in this way allowing tho handle a varying number
       of atoms.
       Only insertions are supported here, but deletions can be achieved by
       taking the negative of the change in structure factors. Translations and
       rotations can be achieved by combining a deletion and an insertion.
       Note that flexible cells are NOT supported => TODO check this
    '''
    def __init__(self, cell, alpha, gcut, pos=None, charges=None, dielectric=1.0,log=None, timer=None):
        '''
            **Arguments:**

            cell
                An instance of the ``Cell`` class.

            alpha
                The alpha parameter in the Ewald summation method.

            gcut
                The cutoff in reciprocal space.

            **Optional arguments:**

            pos
                A [Nx3] Numpy array, providing the coordinates of the atoms
                that are originally present.

            charges
                A [N] Numpy array, providing the charges of the atoms that are
                originally present.

           dielectric
                The scalar relative permittivity of the system.
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        '''
        # Dummy attributes to keep things consistent with ForcePart,
        # these are not actually used.
        self.name = 'ewald_reciprocal_interaction'
        self.energy = 0.0
        self.gpos = np.zeros((0, 3), float)
        self.vtens = np.zeros((3, 3), float)
        if not cell.nvec == 3:
            raise TypeError('The system must have a 3D periodic cell.')
        self.cell = cell
        # Store the original rvecs. If these would change, we need to
        # reinitialize
        self.rvecs0 = self.cell.rvecs.copy()
        self.alpha = alpha
        self.gcut = gcut
        self.dielectric = dielectric
        if log is None:
            from yaff.log import log
        self.log=log
        if timer is None:
            from yaff.log import timer
        self.timer=timer
        self.initialize()
        # Compute the structure factors if an initial configuration is
        # provided.
        if pos is not None:
            assert charges is not None
            self.compute_structurefactors(pos, charges, self.cosfacs, self.sinfacs)
        if self.log.do_medium:
            with self.log.section('EWIINIT'):
                self.log('Ewald Reciprocal interactions')
                self.log.hline()
                self.log('  alpha:             %s' % self.log.invlength(self.alpha))
                self.log('  gcut:              %s' % self.log.invlength(self.gcut))
                self.log.hline()

    def update_gmax(self):
        '''This routine must be called after the attribute self.gmax is modified.'''
        self.gmax = np.ceil(self.gcut/self.cell.gspacings-0.5).astype(int)
        if self.log.do_debug:
            with self.log.section('EWALDI'):
                self.log('gmax a,b,c   = %i,%i,%i' % tuple(self.gmax))

    def initialize(self):
        # Prepare the prefactors \frac{e^{-\frac{k^2}{4\alpha^2}}}{k^2}
        self.update_gmax()
        self.prefactors = np.zeros((2*self.gmax[0]+1,2*self.gmax[1]+1,self.gmax[2]+1))
        compute_ewald_prefactors(self.cell, self.alpha, self.gmax, self.gcut,
                self.prefactors)
        # Prepare the structure factors
        self.cosfacs = np.zeros(self.prefactors.shape)
        self.sinfacs = np.zeros(self.prefactors.shape)
        self.rvecs0 = self.cell.rvecs.copy()

    def compute_structurefactors(self, pos, charges, cosfacs, sinfacs):
        '''Compute the structure factors

           .. math:: \Delta S(\mathbf{k}) = \sum_{i=1}^{M} q_i \
                 e^{j\mathbf{k}\cdot\mathbf{r}_i}

           for the given coordinates and charges. The resulting real part is
           ADDED to cosfacs, the resulting imaginary part is ADDED to sinfacs.
        '''
        with self.timer.section('Ew.reci.SF'):
            if not np.all(self.cell.rvecs==self.rvecs0):
                if self.log.do_medium:
                    with self.log.section('EWALDI'):
                        self.log('Cell change detected, reinitializing')
                self.initialize()
            compute_ewald_structurefactors(pos, charges, self.cell, self.alpha,
                self.gmax, self.gcut, cosfacs, sinfacs)

    def compute_deltae(self, cosfacs, sinfacs):
        '''Compute the energy difference arising if the provided structure
           factors would be added to the current structure factors
        '''
        with self.timer.section('Ew.reci.int.'):
            e = compute_ewald_deltae(self.prefactors, cosfacs, self.cosfacs,
                 sinfacs, self.sinfacs)
        return e/self.dielectric

    def insertion_energy(self, pos, charges, cosfacs=None, sinfacs=None, sign=1):
        '''
        Compute the energy difference if atoms with given coordinates and
        charges are added to the systems. By setting sign to -1, the energy
        difference for removal of those atoms is returned.

            **Arguments:**

            pos
                [Nx3] NumPy array specifying the coordinates

            charges
                [N] NumPy array speficying the charges

            **Optional arguments:**

            cosfacs
                NumPy array with the same shape as the prefactors.
                If not provided, a new array will be created.
                If provided, existing entries will be zerod at the start,
                and contain cosine structure factors of the atoms at the end.

            sinfacs
                NumPy array with the same shape as the prefactors.
                If not provided, a new array will be created.
                If provided, existing entries will be zerod at the start.
                and contain sine structure factors of the atoms at the end.

            sign
                When set to 1, insertion is considered.
                When set to -1, deletion is considered.
        '''
        assert sign in [-1,1]
        if cosfacs is None:
            assert sinfacs is None
            cosfacs = np.zeros(self.prefactors.shape)
            sinfacs = np.zeros(self.prefactors.shape)
        else:
            cosfacs[:] = 0.0
            sinfacs[:] = 0.0
        self.compute_structurefactors(
                pos, charges, cosfacs, sinfacs)
        # We consider a deletion; this means that the structure factors
        # of the considered atoms need to be subtracted from the
        # current system structure factors
        if sign==-1:
            self.cosfacs[:] -= cosfacs
            self.sinfacs[:] -= sinfacs
        return sign*self.compute_deltae(cosfacs, sinfacs)

    def _internal_compute(self, gpos, vtens):
        return 0.0
class ForcePartTIP4(ForcePart):
    '''A force part that implicitly accounts for ghost atoms. Ghost atoms are extra atoms
    in the water molecules to get a better correspondence for the electrostatic interactions
    This part will containt other force parts which are used for the electrostatic interaction.
    These parts use the system_ghost in their calculations, in this system the oxygen atoms are moved to
    the site of the ghost atoms as these oxygen atoms exactly carry the properties of the ghost atoms and
    for the electrostatic part, oxygen atoms themselves have no charge (this is how tip4p works)
    These ghost do not play a role in sampling (verlet, optimization, ) but influence the energy.
    The position of ghost atoms is based on geometric rules.'''
    def __init__(self, system,d_om_rel=0.13194,log=None,timer=None):
        """
           **Arguments**
           system
                An instance of the ``System`` class.

           d_om_rel
               The distance between the ghost atom and the oxygen atom, this lies along the angle bisector of hôh
           **Optional Arguments:**
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        """

        self.system=system
        self.water_atom_indices=self.find_water_atoms()
        self.nwater_atoms=len(self.water_atom_indices)
        self._init_system_ghosts()
        self.nlist=None
        self.gpos_ghosts = np.zeros((self.system_ghosts.pos.shape[0], self.system_ghosts.pos.shape[1]))
        self.vtens = np.zeros((3,3))
        self.d_om_rel=d_om_rel
        self.parts=[]
        self.needs_nlist_update=False
    def _init_system_ghosts(self):
        """

        initialises the system ghosts, which is initially just a copy of the system

        """
        def copy(old):
            if old is None:
                return None
            else:
                return np.copy(old)
        self.system_ghosts=System(
            numbers =  copy(self.system.numbers),
            pos = copy(self.system.pos),
            scopes=copy(self.system.scopes),
            ffatypes=copy(self.system.ffatypes),
            ffatype_ids=copy(self.system.ffatype_ids),
            bonds=copy(self.system.bonds),
            rvecs=copy(self.system.cell.rvecs),
            charges=copy(self.system.charges),
            radii=copy(self.system.radii),
            valence_charges=copy(self.system.valence_charges),
            dipoles=copy(self.system.dipoles),
            radii2=copy(self.system.radii2),
            masses=copy(self.system.masses),
            log=self.log
        )
    def _update_system_ghosts(self):
        '''
        Updates the position of the all atoms in system ghosts based on the position of the system
        also updates the rvecs.

        '''
        #Voorlopig update ik alleen pos en rvecs, voor ei is dit genoeg normaal? mss beter bonds ook?
        self.system_ghosts.pos[:]=self.system.pos[:]
        ghost_pos=self.find_ghost_pos()
        self.system_ghosts.pos[self.water_atom_indices[:,0]]=ghost_pos
        self.system_ghosts.cell.update_rvecs(self.system.cell.rvecs)
    def add_part(self,part):
        """
        Add an electrostatic part to the forcepart
        No electrostatic interactions should be calculated outside this forcepart
        Parameters
        ----------
        part : an instance of a sublcass of ``ForcePart``
            These subclasses need to be all electrostatic interactions
        Raises
        ------
        ValueError
            Every part needs to occur at most once.


        """
        self.parts.append(part)
        # Make the parts also accessible as simple attributes.
        name = 'part_%s' % part.name
        if name in self.__dict__:
            raise ValueError('The part %s occurs twice in the tip4p force part.' % name)
        self.__dict__[name] = part
    def find_ghost_pos(self):
        raise NotImplementedError("use subclasses")
    def write_ghost_gpos(self,gpos_ghosts, vtens_ghosts):
        raise NotImplementedError("use subclasses")
    def find_water_atoms(self):
        """
        Determines the indices of the water molecules

        Returns
        -------
        water_atom_indices : [Nx3] NumPy array of ints
            N is the number of water molecules
            Every row of the array contains the indices of the water atoms, O, H and H respectively

        """
        O_indices=self.system.get_indexes( "8&=2%1")
        water_atom_indices=np.zeros((len(O_indices),3),dtype=int)
        for i,index in enumerate(O_indices):
            water_atom_indices[i,0]+=index
            H_indices=self.system.neighs1[index]
            for j,H_index in enumerate(H_indices):
                water_atom_indices[i,j+1]+=H_index
        return water_atom_indices
    def _internal_compute(self, gpos, vtens):
        with self.timer.section('TIP4'):
            #Write a way to only update in cases when it is necessary
            self._update_system_ghosts()
            self.nlist.update()
            result = sum([part.compute(gpos, vtens) for part in self.parts])
            if not ((gpos is None) and (vtens is None)):
                if gpos is not None and np.isnan(gpos).any():
                    raise ValueError('Some gpos element(s) is/are not-a-number (nan).')
                if gpos is not None:
                    self.write_ghost_gpos(gpos, vtens)
            if (gpos is None) and (vtens is not None):
                raise NotImplementedError("Cannot compute vtens without gpos")


        return result
class ForcePartTIP4P(ForcePartTIP4):
    '''A force part that implicitly accounts for ghost atoms. Ghost atoms are extra atoms
    in the water molecules to get a better correspondence for the electrostatic interactions
    This part will containt other force parts which are used for the electrostatic interaction.
    These parts use the system_ghost in their calculations, in this system the oxygen atoms are moved to
    the site of the ghost atoms as these oxygen atoms exactly carry the properties of the ghost atoms and
    for the electrostatic part, oxygen atoms themselves have no charge (this is how tip4p works)
    These ghost do not play a role in sampling (verlet, optimization, ) but influence the energy.
    The position of ghost atoms is based on geometric rules.'''
    def __init__(self, system,d_om_rel=0.13194,log=None, timer=None):
        """
           **Arguments**
           system
                An instance of the ``System`` class.

           d_om_rel
               The distance between the ghost atom and the oxygen atom, this lies along the angle bisector of hôh
           **Optional Arguments**
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        """
        ForcePart.__init__(self, 'Tip_4P', system,log=log, timer=timer)
        ForcePartTIP4.__init__(self,system)
        self.d_om_rel=d_om_rel
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('TIP4P Force part with %d ghost atoms and %d real atoms.' % (len(self.water_atom_indices), self.system.natom))

    def find_ghost_pos(self):
        '''
        Find ghost site position of TIP4P water based on position of other atoms in molecule
        Assumes that atoms are always order,d_om_rel=0.1319ed as O-H-H-M
            r_M = r_O + d_OM^rel/2 * [ (1+d02/d01)*r01 + (1+d01/d02)*r02 ]
        '''
        ghosts_positions=[]
        for water_molecule in self.water_atom_indices:
            # Vector pointing from O to H1
            r01 = self.system.pos[water_molecule[1]] - self.system.pos[water_molecule[0]]
            self.system.cell.mic(r01)
            d01 = np.linalg.norm(r01)
            # Vector pointing from O to H2
            r02 = self.system.pos[water_molecule[2]] - self.system.pos[water_molecule[0]]
            self.system.cell.mic(r02)
            d02 = np.linalg.norm(r02)
            # Set M position
            M=self.system.pos[water_molecule[0]] + 0.5*self.d_om_rel*((1.0+d02/d01)*r01 + (1.0+d01/d02)*r02)
            self.system.cell.mic(M)
            ghosts_positions.append(M)
        return np.array(ghosts_positions)
    def write_ghost_gpos(self,gpos_ghosts, vtens_ghosts):
        """
        Finds the gradients and the virial tensor of the system based on the gradients and virial tensor of the ghost system
        then overwrites the arguments to represent the system
        Parameters

           gpos_ghosts
                gpos of the ghost system
           vtens_ghosts
                vtens of the ghost system
        """
        for water_molecule in self.water_atom_indices:
            # Vector pointing from O to H1
            r01 = self.system.pos[water_molecule[1]] - self.system.pos[water_molecule[0]]
            self.system.cell.mic(r01)
            d01 = np.linalg.norm(r01)
            # Vector pointing from O to H2
            r02 = self.system.pos[water_molecule[2]] - self.system.pos[water_molecule[0]]
            self.system.cell.mic(r02)
            d02 = np.linalg.norm(r02)
            # Partial derivatives of M positions
            pdiff_01 = gpos_ghosts[water_molecule[0],:]*(1.0+d02/d01) - r01*np.dot(gpos_ghosts[water_molecule[0],:],(d02/d01/d01/d01*r01-r02/d01/d02))
            pdiff_02 = gpos_ghosts[water_molecule[0],:]*(1.0+d01/d02) - r02*np.dot(gpos_ghosts[water_molecule[0],:],(d01/d02/d02/d02*r02-r01/d02/d01))
            # Apply chain rule
            if vtens_ghosts is not None:
                r_mo = 0.5*self.d_om_rel*((1.0+d02/d01)*r01 + (1.0+d01/d02)*r02)
                vtens_ghosts[:] -= np.outer(gpos_ghosts[water_molecule[0],:],r_mo)
                vtens_ghosts[:] += np.outer(0.5*self.d_om_rel*pdiff_01,r01)
                vtens_ghosts[:] += np.outer(0.5*self.d_om_rel*pdiff_02,r02)
            gpos_ghosts[water_molecule[0],:] -= 0.5*self.d_om_rel*pdiff_01
            gpos_ghosts[water_molecule[0],:] -= 0.5*self.d_om_rel*pdiff_02
            gpos_ghosts[water_molecule[1],:] += 0.5*self.d_om_rel*pdiff_01
            gpos_ghosts[water_molecule[2],:] += 0.5*self.d_om_rel*pdiff_02
class ForcePartQTIP4P(ForcePartTIP4):
    '''A force part that implicitly accounts for ghost atoms. Ghost atoms are extra atoms
    in the water molecules to get a better correspondence for the electrostatic interactions
    This part will containt other force parts which are used for the electrostatic interaction.
    These parts use the system_ghost in their calculations, in this system the oxygen atoms are moved to
    the site of the ghost atoms as these oxygen atoms exactly carry the properties of the ghost atoms and
    for the electrostatic part, oxygen atoms themselves have no charge (this is how tip4p works)
    These ghost do not play a role in sampling (verlet, optimization, ) but influence the energy.
    The position of ghost atoms is based on geometric rules.'''
    def __init__(self, system,gamma=0.73612,log=None, timer=None):
        """
           **Arguments**
           system
                An instance of the ``System`` class.

           d_om_rel
               The distance between the ghost atom and the oxygen atom, this lies along the angle bisector of hôh
           **OPtional Arguments**
           log
                A Screenlog object can be passed locally
                if None, the global log is used

           timer
                A TimerGroup object can be passed locally
                if None, the global timer is used
        """
        ForcePart.__init__(self, 'Tip_4P', system,log=log, timer=timer)
        ForcePartTIP4.__init__(self,system)
        self.gamma=gamma
        if self.log.do_medium:
            with self.log.section('FPINIT'):
                self.log('TIP4P Force part with %d ghost atoms and %d real atoms.' % (len(self.water_atom_indices), self.system.natom))

    def find_ghost_pos(self):
        '''
        Find ghost site position of TIP4P water based on position of other atoms in molecule
        Assumes that atoms are always order, as O-H-H-M
            r_M = r_O + d_OM^rel/2 * [ (1+d02/d01)*r01 + (1+d01/d02)*r02 ]
        '''
        ghosts_positions=[]
        for water_molecule in self.water_atom_indices:
            # Vector pointing from O to H1
            r01 = self.system.pos[water_molecule[1]] - self.system.pos[water_molecule[0]]
            self.system.cell.mic(r01)
            r02 = self.system.pos[water_molecule[2]] - self.system.pos[water_molecule[0]]
            self.system.cell.mic(r02)
            # Set M position
            M=self.system.pos[water_molecule[0]] + 0.5*(1-self.gamma)*(r01 + r02)
            self.system.cell.mic(M)
            ghosts_positions.append(M)
        return np.array(ghosts_positions)
    def write_ghost_gpos(self,gpos_ghosts, vtens_ghosts):
        """
        Finds the gradients and the virial tensor of the system based on the gradients and virial tensor of the ghost system
        then overwrites the arguments to represent the system
        Parameters

           gpos_ghosts
                gpos of the ghost system
           vtens_ghosts
                vtens of the ghost system
        """
        for water_molecule in self.water_atom_indices:
            # Vector pointing from O to H1
            r01 = self.system.pos[water_molecule[1]] - self.system.pos[water_molecule[0]]
            self.system.cell.mic(r01)
            # Vector pointing from O to H2
            r02 = self.system.pos[water_molecule[2]] - self.system.pos[water_molecule[0]]
            self.system.cell.mic(r02)
            # Partial derivatives of M positions

            if vtens_ghosts is not None:
                r_mo = 0.5*(1-self.gamma)*(r01 + r02)
                vtens_ghosts[:] -= np.outer(gpos_ghosts[water_molecule[0],:],r_mo)
                vtens_ghosts[:] += np.outer(0.5*(1-self.gamma)*gpos_ghosts[water_molecule[0],:],r01)
                vtens_ghosts[:] += np.outer(0.5*(1-self.gamma)*gpos_ghosts[water_molecule[0],:],r02)
            gpos_ghosts[water_molecule[1],:] += 0.5*(1-self.gamma)*gpos_ghosts[water_molecule[0],:]
            gpos_ghosts[water_molecule[2],:] += 0.5*(1-self.gamma)*gpos_ghosts[water_molecule[0],:]
            gpos_ghosts[water_molecule[0],:] *=self.gamma

def check_nlow_nhigh(system, nlow, nhigh):
    if nlow < 0:
        raise ValueError('nlow must be positive.')
    if nlow > system.natom:
        raise ValueError('nlow must not be larger than system.natom')
    if nhigh == -1: nhigh = system.natom
    if nhigh < nlow:
        raise ValueError('nhigh must not be smaller than nlow')
    if nhigh > system.natom:
        raise ValueError('nhigh must not be larger than system.natom')
    return nlow, nhigh
