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
'''Equations of state'''


from __future__ import division

import numpy as np
from scipy.optimize import newton as newton_opt # Avoid clash with newton from molmod.units
import pkg_resources

from molmod import boltzmann, planck, amu, pascal, kelvin

import yaff


__all__ = [
    'IdealGas', 'vdWEOS', 'PREOS',
]


class EOS(object):
    def __init__(self, mass=0.0):
        self.mass = mass

    def calculate_fugacity(self, T, P):
        """
           Evaluate the excess chemical potential at given external conditions

           **Arguments:**

           T
                Temperature

           P
                Pressure

           **Returns:**

           f
                The fugacity
        """
        mu, Pref = self.calculate_mu_ex(T, P)
        fugacity = np.exp( mu/(boltzmann*T) )*Pref
        return fugacity

    def calculate_mu(self, T, P):
        """
           Evaluate the chemical potential at given external conditions

           **Arguments:**

           T
                Temperature

           P
                Pressure

           **Returns:**

           mu
                The chemical potential
        """
        # Excess part
        mu, Pref = self.calculate_mu_ex(T,P)
        # Ideal gas contribution to chemical potential
        assert self.mass!=0.0
        lambd = 2.0*np.pi*self.mass*boltzmann*T/planck**2
        mu0 = -boltzmann*T*np.log( boltzmann*T/Pref*lambd**1.5)
        return mu0+mu

    def get_Pref(self, T, P0, deviation=1e-3):
        """
           Find a reference pressure at the given temperature for which the
           fluidum is nearly ideal.


           **Arguments:**

           T
                Temperature

           P
                Pressure

           **Optional arguments:**

           deviation
                When the compressibility factor Z deviates less than this from
                1, ideal gas behavior is assumed.
        """
        Pref = P0
        for i in range(100):
            rhoref = self.calculate_rho(T, Pref)
            Zref = Pref/rhoref/boltzmann/T
            # Z close to 1.0 means ideal gas behavior
            if np.abs(Zref-1.0)>deviation:
                Pref /= 2.0
            else: break
        if np.abs(Zref-1.0)>deviation:
            raise ValueError("Failed to find pressure where the fluidum is ideal-gas like, check input parameters")
        return Pref


class IdealGas(EOS):
    """The ideal gas equation of state"""
    def calculate_mu_ex(self, T, P):
        mu = 0.0
        Pref = P
        return mu, Pref

    def calculate_rho(self, T, P, rho0=None):
        """
           Calculate the particle density at given external conditions

           **Arguments:**

           T
                Temperature

           P
                Pressure

           **Optional arguments:**

           rho0
                Initial guess for the density, not used for this specific EOS.

           **Returns:**

           rho
                The particle density
        """
        return P/boltzmann/T


class vdWEOS(EOS):
    """The van der Waals equation of state"""
    def __init__(self, a, b, mass=0.0):
        self.a = a
        self.b = b
        self.mass = mass

    def polynomial(self, rho, T, P):
        poly = -self.a*self.b*rho**3
        poly += self.a*rho**2
        poly -= self.b*P*rho
        poly -= boltzmann*T*rho
        poly += P
        return poly

    def calculate_rho(self, T, P, rho0=None):
        """
           Calculate the particle density at given external conditions

           **Arguments:**

           T
                Temperature

           P
                Pressure

           **Optional arguments:**

           rho0
                Initial guess for the density. If not provided, an initial
                guess based on the ideal gas law is used.

           **Returns:**

           rho
                The particle density
        """
        if rho0 is None:
            rho0 = P/boltzmann/T
        return newton_opt(self.polynomial, rho0, args=(T,P), tol=1e-10)

    def calculate_mu_ex(self, T, P):
        """
           Evaluate the excess chemical potential at given external conditions

           **Arguments:**

           T
                Temperature

           P
                Pressure

           **Returns:**

           mu
                The excess chemical potential

           Pref

                The pressure at which the reference chemical potential was calculated
        """
        # Find a reference pressure at the given temperature for which the fluidum
        # is nearly ideal
        Pref = self.get_Pref(T, P)
        # Find zero of polynomial expression to get the density
        rho = self.calculate_rho(T, P)
        # Add contributions to chemical potential at requested pressure
        mu = boltzmann*T*( np.log(self.b*rho/(1.0-self.b*rho)) + self.b*rho/(1.0-self.b*rho))
        mu -= 2.0*self.a*rho
        mu -= boltzmann*T*np.log(Pref*self.b/boltzmann/T)
        return mu, Pref


class PREOS(EOS):
    """The Peng-Robinson equation of state"""
    def __init__(self, Tc, Pc, omega, mass=0.0, phase="vapour",log=None):
        """
           The Peng-Robinson EOS gives a relation between pressure, volume, and
           temperature with parameters based on the critical pressure, critical
           temperature and acentric factor.

           **Arguments:**

           Tc
                The critical temperature of the species

           Pc
                The critical pressure of the species

           omega
                The acentric factor of the species

           **Optional arguments:**

           mass
                The mass of one molecule of the species. Some properties can be
                computed without this, so it is an optional argument

           phase
                Either "vapour" or "liquid". If both phases coexist at certain
                conditions, properties for the selected phase will be reported.
           log 
                A Screenlog object can be passed locally
                if None, the global log is used.
        """
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.mass = mass
        self.phase = phase
        # Some parameters derived from the input parameters
        self.a = 0.457235 * self.Tc**2 / self.Pc
        self.b = 0.0777961 * self.Tc / self.Pc
        self.kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2
        if log is None:
            from yaff.log import log
        self.log=log

    @classmethod
    def from_name(cls, compound,log=None):
        """
           Initialize a Peng-Robinson EOS based on the name of the compound.
           Only works if the given compound name is included in
           'yaff/data/critical_acentric.csv'
        """
        # Read the data file containing parameters for a number of selected compounds
        fn = pkg_resources.resource_filename(yaff.__name__, 'data/critical_acentric.csv')
        dtype=[('compound','S20'),('mass','f8'),('Tc','f8'),('Pc','f8'),('omega','f8'),]
        data = np.genfromtxt(fn, dtype=dtype, delimiter=',')
        # Select requested compound
        if not compound.encode('utf-8') in data['compound']:
            raise ValueError("Could not find data for %s in file %s"%(compound,fn))
        index = np.where( compound.encode('utf-8') == data['compound'] )[0]
        assert index.shape[0]==1
        mass = data['mass'][index[0]]*amu
        Tc = data['Tc'][index[0]]*kelvin
        Pc = data['Pc'][index[0]]*1e6*pascal
        omega = data['omega'][index[0]]
        return cls(Tc, Pc, omega, mass=mass,log=log)

    def set_conditions(self, T, P):
        """
           Set the parameters that depend on T and P

           **Arguments:**

           T
                Temperature

           P
                Pressure
        """
        self.Tr = T / self.Tc  # reduced temperature
        self.alpha = (1 + self.kappa * (1 - np.sqrt(self.Tr)))**2
        self.A = self.a * self.alpha * P / T**2
        self.B = self.b * P / T

    def polynomial(self, Z):
        """
           Evaluate the polynomial form of the Peng-Robinson equation of state
           If returns zero, the point lies on the PR EOS curve

           **Arguments:**

           Z
                Compressibility factor
        """
        return Z**3 - (1 - self.B) * Z**2 + (self.A - 2*self.B - 3*self.B**2) * Z - (
                self.A * self.B - self.B**2 - self.B**3)

    def polynomial_roots(self):
        """
            Find the real roots of the polynomial form of the Peng-Robinson
            equation of state
        """
        a = - (1 - self.B)
        b = self.A - 2*self.B - 3*self.B**2
        c = - (self.A * self.B - self.B**2 - self.B**3)
        Q = (a**2-3*b)/9
        R = (2*a**3-9*a*b+27*c)/54
        M = R**2-Q**3
        if M>0:
            S = np.cbrt(-R+np.sqrt(M))
            T = np.cbrt(-R-np.sqrt(M))
            Z = S+T-a/3
        else:
            theta = np.arccos(R/np.sqrt(Q**3))
            x1 = -2.0*np.sqrt(Q)*np.cos(theta/3)-a/3
            x2 = -2.0*np.sqrt(Q)*np.cos((theta+2*np.pi)/3)-a/3
            x3 = -2.0*np.sqrt(Q)*np.cos((theta-2*np.pi)/3)-a/3
            solutions = np.array([x1,x2,x3])
            solutions = solutions[solutions>0.0]
            if self.phase=='vapour':
                Z = np.amax(solutions)
            elif self.phase=='liquid':
                Z = np.amin(solutions)
            else: raise NotImplementedError
            if self.log.do_high:
                self.log("Found 3 solutions for Z (%f,%f,%f), meaning that two "
                    "phases coexist. Returning Z=%f, corresponding to the "
                    "%s phase"%(x1,x2,x3,Z,self.phase))
        return Z

    def calculate_rho(self, T, P):
        """
           Calculate the particle density at given external conditions

           **Arguments:**

           T
                Temperature

           P
                Pressure

           **Returns:**

           rho
                The particle density
        """
        self.set_conditions(T, P)
        Z = self.polynomial_roots()
        return P/Z/boltzmann/T

    def calculate_mu_ex(self, T, P):
        """
           Evaluate the excess chemical potential at given external conditions

           **Arguments:**

           T
                Temperature

           P
                Pressure

           **Returns:**

           mu
                The excess chemical potential

           Pref

                The pressure at which the reference chemical potential was calculated
        """
        # Find a reference pressure at the given temperature for which the fluidum
        # is nearly ideal
        Pref = self.get_Pref(T, P)
        # Find compressibility factor using rho
        rho = self.calculate_rho(T, P)
        Z = P/rho/boltzmann/T
        # Add contributions to chemical potential at requested pressure
        mu = Z - 1 - np.log(Z - self.B) - self.A / np.sqrt(8) / self.B * np.log(
                    (Z + (1 + np.sqrt(2)) * self.B) / (Z + (1 - np.sqrt(2)) * self.B))
        mu += np.log(P/Pref)
        mu *= T*boltzmann
        return mu, Pref
