# YAFF is yet another force-field code
# Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
# for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights
# reserved unless otherwise stated.
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


from yaff.sampling.iterative import Hook


__all__ = ['HDF5Writer', 'XYZWriter']


class HDF5Writer(Hook):
    def __init__(self, f, start=0, step=1):
        """
           **Argument:**

           f
                A h5py.File object to write the trajectory to.

           **Optional arguments:**

           start
                The first iteration at which this hook should be called.

           step
                The hook will be called every `step` iterations.
        """
        self.f = f
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        if 'system' not in self.f:
            self.dump_system(iterative.ff.system)
        if 'trajectory' not in self.f:
            self.init_trajectory(iterative)
        tgrp = self.f['trajectory']
        # Get the number of rows written so far. It may seem redundant to have
        # the number of rows stored as an attribute, while each dataset also
        # has a shape from which the number of rows can be determined. However,
        # this helps to keep the dataset sane in case this write call got
        # interrupted in the loop below. Only when the last line below is
        # executed, the data from this iteration is officially written.
        row = tgrp.attrs['row']
        for key, item in iterative.state.iteritems():
            if len(item.shape) > 0 and min(item.shape) == 0:
                continue
            if item.dtype is type(None):
                continue
            ds = tgrp[key]
            if ds.shape[0] <= row:
                # do not over-allocate. hdf5 works with chunks internally.
                ds.resize(row+1, axis=0)
            ds[row] = item.value
        tgrp.attrs['row'] += 1

    def dump_system(self, system):
        sgrp = self.f.create_group('system')
        sgrp.create_dataset('numbers', data=system.numbers)
        sgrp.create_dataset('pos', data=system.pos)
        if system.scopes is not None:
            sgrp.create_dataset('scopes', data=system.scopes, dtype='a22')
            sgrp.create_dataset('scope_ids', data=system.scope_ids)
        if system.ffatypes is not None:
            sgrp.create_dataset('ffatypes', data=system.ffatypes, dtype='a22')
            sgrp.create_dataset('ffatype_ids', data=system.ffatype_ids)
        if system.bonds is not None:
            sgrp.create_dataset('bonds', data=system.bonds)
        if system.cell.nvec > 0:
            sgrp.create_dataset('rvecs', data=system.cell.rvecs)
        if system.charges is not None:
            sgrp.create_dataset('charges', data=system.charges)
        if system.masses is not None:
            sgrp.create_dataset('masses', data=system.masses)

    def init_trajectory(self, iterative):
        tgrp = self.f.create_group('trajectory')
        for key, item in iterative.state.iteritems():
            if len(item.shape) > 0 and min(item.shape) == 0:
                continue
            if item.dtype is type(None):
                continue
            maxshape = (None,) + item.shape
            shape = (0,) + item.shape
            dset = tgrp.create_dataset(key, shape, maxshape=maxshape, dtype=item.dtype)
            for name, value in item.iter_attrs(iterative):
               tgrp.attrs[name] = value
        tgrp.attrs['row'] = 0


class XYZWriter(Hook):
    def __init__(self, fn_xyz, select=None, start=0, step=1):
        """
           **Argument:**

           fn_xyz
                A filename to write the XYZ trajectory too.

           **Optional arguments:**

           select
                A list of atom indexes that should be written to the trajectory
                output. If not given, all atoms are included.

           start
                The first iteration at which this hook should be called.

           step
                The hook will be called every `step` iterations.
        """
        self.fn_xyz = fn_xyz
        self.select = select
        self.xyz_writer = None
        Hook.__init__(self, start, step)

    def __call__(self, iterative):
        from molmod import angstrom
        if self.xyz_writer is None:
            from molmod.periodic import periodic
            from molmod.io import XYZWriter
            numbers = iterative.ff.system.numbers
            if self.select is None:
                symbols = [periodic[n].symbol for n in numbers]
            else:
                symbols = [periodic[numbers[i]].symbol for i in self.select]
            self.xyz_writer = XYZWriter(self.fn_xyz, symbols)
        rvecs = iterative.ff.system.cell.rvecs.copy()
        rvecs_string = " ".join([str(x[0]) for x in rvecs.reshape((-1,1))])
        title = '%7i E_pot = %.10f    %s' % (iterative.counter, iterative.epot, rvecs_string)
        if self.select is None:
            pos = iterative.ff.system.pos
        else:
            pos = iterative.ff.system.pos[self.select]
        self.xyz_writer.dump(title, pos)
