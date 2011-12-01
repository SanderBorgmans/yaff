// YAFF is yet another force-field code
// Copyright (C) 2008 - 2011 Toon Verstraelen <Toon.Verstraelen@UGent.be>, Center
// for Molecular Modeling (CMM), Ghent University, Ghent, Belgium; all rights
// reserved unless otherwise stated.
//
// This file is part of YAFF.
//
// YAFF is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// YAFF is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, see <http://www.gnu.org/licenses/>
//
// --

#include <math.h>
#include "nlist.h"
#include "cell.h"

// TODO: flatten sign-loop in nlist_build_low by including sign in the status
//       and change sign in the part where the counters of the quintuple loop
//       are updated.

int nlist_build_low(double *pos, double rcut, long *rmax,
                    cell_type *unitcell, long *status,
                    neigh_row_type *neighs, long natom, long nneigh) {

  long a, b, row;
  long *r;
  int update_delta0, image, sign;
  double delta0[3], delta[3], d;

  r = status;
  a = status[3];
  b = status[4];

  update_delta0 = 1;
  image = (r[0] != 0) || (r[1] != 0) || (r[2] != 0);
  row = 0;

  while (row < nneigh-1) {
    if (a >= natom) {
      // Completely done.
      status[5] += row;
      return 1;
    }
    // Avoid adding pairs for which a > b and that match the minimum image
    // convention.
    if (update_delta0) {
      // Compute the relative vector.
      delta0[0] = pos[3*b  ] - pos[3*a  ];
      delta0[1] = pos[3*b+1] - pos[3*a+1];
      delta0[2] = pos[3*b+2] - pos[3*a+2];
      // Subtract the cell vectors as to make the relative vector as short
      // as possible. (This is the minimum image convention.)
      cell_mic(delta0, unitcell);
      // Done updating delta0.
      update_delta0 = 0;
    }
    // Construct delta by adding the appropriate cell vector to delta0
    for (sign=1; sign >= -image; sign -= 2) {
      delta[0] = sign*delta0[0];
      delta[1] = sign*delta0[1];
      delta[2] = sign*delta0[2];
      cell_add_vec(delta, unitcell, r);
      // Compute the distance and store the record if distance is below the rcut.
      d = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
      if (d < rcut) {
        if (sign > 0) {
          (*neighs).a = a;
          (*neighs).b = b;
        } else {
          (*neighs).a = b;
          (*neighs).b = a;
        }
        (*neighs).d = d;
        (*neighs).dx = delta[0];
        (*neighs).dy = delta[1];
        (*neighs).dz = delta[2];
        (*neighs).r0 = r[0];
        (*neighs).r1 = r[1];
        (*neighs).r2 = r[2];
        neighs++;
        row++;
      }
    }
    // Increase the appropriate counters in the quintuple loop.
    if (!nlist_inc_r(unitcell, r, rmax)) {
      update_delta0 = 1;
      image = 0;
      b++;
      if (b >= a) {
        b = 0;
        a++;
      }
    } else {
      image = 1;
    }
  }
  // Exit before the job is done. Keep track of the status. Work will be resumed
  // in a next call.
  status[0] = r[0];
  status[1] = r[1];
  status[2] = r[2];
  status[3] = a;
  status[4] = b;
  status[5] += row;
  return 0;
}


int nlist_inc_r(cell_type *unitcell, long *r, long *rmax) {
  // increment the counters for the periodic images.
  // returns 1 when the counters were incremented succesfully.
  // returns 0 and resets all the counters when the iteration over all cells
  // is complete.

  // Note: Only the central image and half of the neighboring images are
  // considered. This way, one can build neighborlists without any duplicates
  // pairs. The central cell comes first.
  if ((*unitcell).nvec > 0) {
    r[0]++;
    if (r[0] > rmax[0]) {
      r[0] = -rmax[0];
      if ((*unitcell).nvec > 1) {
        r[1]++;
        if (r[1] > rmax[1]) {
          r[1] = -rmax[1];
          if ((*unitcell).nvec > 2) {
            r[2]++;
            if (r[2] > rmax[2]) {
              r[2] = -rmax[2];
              goto endloop;
            }
          } else {
            goto endloop;
          }
        }
      } else {
        goto endloop;
      }
    }
  } else {
    return 0;
  }
  return 1;

endloop:
  // This point is only reached when all relevant neighboring cells are visited.
  // We set the counters back to the central cell
  if ((*unitcell).nvec > 0) r[0] = 0;
  if ((*unitcell).nvec > 1) r[1] = 0;
  if ((*unitcell).nvec > 2) r[2] = 0;
  return 0;
}


void nlist_recompute_low(double *pos, double *pos_old, cell_type* unitcell,
                         neigh_row_type *neighs, long nneigh) {
  long i, a, b;
  int update_delta0;
  long center[3];
  double delta0[3], delta[3], d;

  update_delta0 = 1;
  a = -1;
  b = -1;

  for (i=nneigh-1; i>=0; i--) {
    if ((*neighs).a != a) {
      update_delta0 = 1;
    } else if ((*neighs).b != b) {
      update_delta0 = 1;
    }
    if (update_delta0) {
      a = (*neighs).a;
      b = (*neighs).b;
      // Compute the old relative vector.
      delta0[0] = pos_old[3*b  ] - pos_old[3*a  ];
      delta0[1] = pos_old[3*b+1] - pos_old[3*a+1];
      delta0[2] = pos_old[3*b+2] - pos_old[3*a+2];
      // Compute the cell vectors to be subtracted to bring the old to the MIC
      cell_to_center(delta0, unitcell, center);
      // Compute the new relative vector.
      delta0[0] = pos[3*b  ] - pos[3*a  ];
      delta0[1] = pos[3*b+1] - pos[3*a+1];
      delta0[2] = pos[3*b+2] - pos[3*a+2];
      // Apply the same cell displacement to the new relative vector
      cell_add_vec(delta0, unitcell, center);
      // Done updating delta0.
      update_delta0 = 0;
    }
    // Construct delta by adding the appropriate cell vector to delta0
    delta[0] = delta0[0];
    delta[1] = delta0[1];
    delta[2] = delta0[2];
    cell_add_vec(delta, unitcell, &((*neighs).r0));
    // Compute the distance and store the record if distance is below the rcut.
    d = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
    // Store the new results;
    (*neighs).d = d;
    (*neighs).dx = delta[0];
    (*neighs).dy = delta[1];
    (*neighs).dz = delta[2];
    neighs++;
  }
}
