"""
NERF!
Note that this was designed with compatibility with biotite, NOT biopython!
These two packages use different conventions for where NaNs are placed in dihedrals

References:
https://benjamin-computer.medium.com/protein-loops-in-tensorflow-a-i-bio-part-2-f1d802ef8300
https://www.biotite-python.org/examples/gallery/structure/peptide_assembly.html
"""
import os
from functools import cached_property
from typing import *

import numpy as np
import torch

N_CA_LENGTH = 1.46  # Check, approxiamtely right
CA_C_LENGTH = 1.54  # Check, approximately right
C_N_LENGTH = 1.34  # Check, approximately right

# Taken from initial coords from 1CRN, which is a THR
N_INIT = np.array([17.047, 14.099, 3.625])
CA_INIT = np.array([16.967, 12.784, 4.338])
C_INIT = np.array([15.685, 12.755, 5.133])


class NERFBuilder:
    """
    Builder for NERF
    """

    def __init__(
        self,
        phi_dihedrals: np.ndarray,
        psi_dihedrals: np.ndarray,
        omega_dihedrals: np.ndarray,
        bond_len_n_ca: Union[float, np.ndarray] = N_CA_LENGTH,
        bond_len_ca_c: Union[float, np.ndarray] = CA_C_LENGTH,
        bond_len_c_n: Union[float, np.ndarray] = C_N_LENGTH,  # 0C:1N distance
        bond_angle_n_ca: Union[float, np.ndarray] = 121 / 180 * np.pi,
        bond_angle_ca_c: Union[float, np.ndarray] = 109 / 180 * np.pi,  # aka tau
        bond_angle_c_n: Union[float, np.ndarray] = 115 / 180 * np.pi,
        init_coords: np.ndarray = [N_INIT, CA_INIT, C_INIT],
    ) -> None:
        self.use_torch = False
        if any(
            [
                isinstance(v, torch.Tensor)
                for v in [phi_dihedrals, psi_dihedrals, omega_dihedrals]
            ]
        ):
            self.use_torch = True

        self.phi = phi_dihedrals.squeeze()
        self.psi = psi_dihedrals.squeeze()
        self.omega = omega_dihedrals.squeeze()

        # We start with coordinates for N --> CA --> C so the next atom we add
        # is the next N. Therefore, the first angle we need is the C --> N bond
        self.bond_lengths = {
            ("C", "N"): bond_len_c_n,
            ("N", "CA"): bond_len_n_ca,
            ("CA", "C"): bond_len_ca_c,
        }
        self.bond_angles = {
            ("C", "N"): bond_angle_c_n,
            ("N", "CA"): bond_angle_n_ca,
            ("CA", "C"): bond_angle_ca_c,
        }
        self.init_coords = [c.squeeze() for c in init_coords]
        assert (
            len(self.init_coords) == 3
        ), f"Requires 3 initial coords for N-Ca-C but got {len(self.init_coords)}"
        assert all(
            [c.size == 3 for c in self.init_coords]
        ), "Initial coords should be 3-dimensional"

    @cached_property
    def cartesian_coords(self) -> Union[np.ndarray, torch.Tensor]:
        """Build out the molecule"""
        retval = self.init_coords.copy()
        if self.use_torch:
            retval = [torch.tensor(x, requires_grad=True) for x in retval]

        # The first value of phi at the N terminus is not defined
        # The last value of psi and omega at the C terminus are not defined
        phi = self.phi[1:]
        psi = self.psi[:-1]
        omega = self.omega[:-1]
        dih_angles = (
            torch.stack([psi, omega, phi])
            if self.use_torch
            else np.stack([psi, omega, phi])
        ).T
        assert (
            dih_angles.shape[1] == 3
        ), f"Unexpected dih_angles shape: {dih_angles.shape}"

        for i in range(dih_angles.shape[0]):
            # for i, (phi, psi, omega) in enumerate(
            #     zip(self.phi[1:], self.psi[:-1], self.omega[:-1])
            # ):
            dih = dih_angles[i]
            # Procedure for placing N-CA-C
            # Place the next N atom, which requires the C-N bond length/angle, and the psi dihedral
            # Place the alpha carbon, which requires the N-CA bond length/angle, and the omega dihedral
            # Place the carbon, which requires the the CA-C bond length/angle, and the phi dihedral
            for j, bond in enumerate(self.bond_lengths.keys()):
                coords = place_dihedral(
                    retval[-3],
                    retval[-2],
                    retval[-1],
                    bond_angle=self._get_bond_angle(bond, i),
                    bond_length=self._get_bond_length(bond, i),
                    torsion_angle=dih[j],
                    use_torch=self.use_torch,
                )
                retval.append(coords)

        if self.use_torch:
            return torch.stack(retval)
        return np.array(retval)

    @cached_property
    def centered_cartesian_coords(self) -> Union[np.ndarray, torch.Tensor]:
        """Returns the centered coords"""
        means = self.cartesian_coords.mean(axis=0)
        return self.cartesian_coords - means

    def _get_bond_length(self, bond: Tuple[str, str], idx: int):
        """Get the ith bond distance"""
        v = self.bond_lengths[bond]
        if isinstance(v, float):
            return v
        return v[idx]

    def _get_bond_angle(self, bond: Tuple[str, str], idx: int):
        """Get the ith bond angle"""
        v = self.bond_angles[bond]
        if isinstance(v, float):
            return v
        return v[idx]


def place_dihedral(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_angle: float,
    bond_length: float,
    torsion_angle: float,
    use_torch: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Place the point d such that the bond angle, length, and torsion angle are satisfied
    with the series a, b, c, d.
    """
    assert a.shape == b.shape == c.shape
    assert a.shape[-1] == b.shape[-1] == c.shape[-1] == 3

    if not use_torch:
        unit_vec = lambda x: x / np.linalg.norm(x, axis=-1)
        cross = lambda x, y: np.cross(x, y, axis=-1)
    else:
        ensure_tensor = (
            lambda x: torch.tensor(x, requires_grad=False).to(a.device)
            if not isinstance(x, torch.Tensor)
            else x.to(a.device)
        )
        a, b, c, bond_angle, bond_length, torsion_angle = [
            ensure_tensor(x) for x in (a, b, c, bond_angle, bond_length, torsion_angle)
        ]
        unit_vec = lambda x: x / torch.linalg.norm(x, dim=-1, keepdim=True)
        cross = lambda x, y: torch.linalg.cross(x, y, dim=-1)

    ab = b - a
    bc = unit_vec(c - b)
    n = unit_vec(cross(ab, bc))
    nbc = cross(n, bc)

    if not use_torch:
        m = np.stack([bc, nbc, n], axis=-1)
        d = np.stack(
            [
                -bond_length * np.cos(bond_angle),
                bond_length * np.cos(torsion_angle) * np.sin(bond_angle),
                bond_length * np.sin(torsion_angle) * np.sin(bond_angle),
            ],
            axis=a.ndim - 1,
        )
        d = m.dot(d)
    else:
        m = torch.stack([bc, nbc, n], dim=-1)
        d = torch.stack(
            [
                -bond_length * torch.cos(bond_angle),
                bond_length * torch.cos(torsion_angle) * torch.sin(bond_angle),
                bond_length * torch.sin(torsion_angle) * torch.sin(bond_angle),
            ],
            dim=a.ndim - 1,
        ).type(m.dtype)
        d = torch.matmul(m, d).squeeze()

    return d + c


def nerf_build_batch(
    phi: torch.Tensor,
    psi: torch.Tensor,
    omega: torch.Tensor,
    bond_angle_n_ca_c: torch.Tensor = 0,  # theta1
    bond_angle_ca_c_n: torch.Tensor = 0,  # theta2
    bond_angle_c_n_ca: torch.Tensor = 0,  # theta3
    bond_len_n_ca: Union[float, torch.Tensor] = N_CA_LENGTH,
    bond_len_ca_c: Union[float, torch.Tensor] = CA_C_LENGTH,
    bond_len_c_n: Union[float, torch.Tensor] = C_N_LENGTH,  # 0C:1N distance
) -> torch.Tensor:
    """
    Build out a batch of phi, psi, omega values. Returns the 3D coordinates
    in Cartesian space with the shape (batch, length, atoms, xyz).
    """
    assert phi.ndim == psi.ndim == omega.ndim == 2  # batch, seq
    assert phi.shape == psi.shape == omega.shape

    # Use Sergey's code instead of the original implementation
    batch_dih = torch.stack([phi, psi, omega], dim=-1)
    
    # The returned shape is (batch, length, atoms=(N,CA,C,CB,O), coords=(x,y,z))
    coords = dih_to_coord(batch_dih, add_ob=False, requires_grad=True)

    return coords


# Code adapted from Sergey Ovchinnikov
def to_len(a, b):
    """
    ====================================================================
    Given coordinates a-b, return length or distance.
    ====================================================================
    """
    return torch.sqrt(torch.sum((a - b) ** 2, dim=-1) + 1e-8)


def to_ang(a, b, c):
    """
    ====================================================================
    Given coordinates a-b-c, return angle.
    ====================================================================
    """
    N = lambda x: torch.sqrt(torch.sum(x ** 2, dim=-1) + 1e-8)
    ba = b - a
    bc = b - c
    ang = torch.acos(torch.sum(ba * bc, dim=-1) / (N(ba) * N(bc)))
    return ang


def to_dih(a, b, c, d):
    """
    ====================================================================
    Given coordinates a-b-c-d, return dihedral.
    ====================================================================
    """
    D = lambda x, y: torch.sum(x * y, dim=-1)
    N = lambda x: torch.nn.functional.normalize(x, dim=-1)
    bc = N(b - c)
    n1 = torch.cross(N(a - b), bc)
    n2 = torch.cross(bc, N(c - d))
    x = D(n1, n2)
    y = D(torch.cross(n1, bc), n2)
    dih = torch.atan2(y, x)
    return dih


def coord_to_dih(coord):
    """
    ====================================================================
    Given coordinates (N,CA,C), return dihedrals (phi,psi,omega).
    
    input:  (batch,length,atoms=(N,CA,C),coords=(x,y,z))
    output: (batch,length,dihedrals=(phi,psi,omega))
    ====================================================================
    """
    batch_size = coord.shape[0]
    length = coord.shape[1]
    
    X = coord[:, :, :3].reshape(batch_size, -1, 3)
    a, b, c, d = X[:, :-3], X[:, 1:-2], X[:, 2:-1], X[:, 3:]
    dih = to_dih(a, b, c, d)
    
    # add zero (psi) at start
    # add zero (phi,omega) at end
    # reshape to (batch,length,(phi,psi,omega))
    dih = torch.nn.functional.pad(dih, (1, 2))
    dih = dih.reshape(batch_size, -1, 3)
    return dih


def extend(a, b, c, L, A, D):
    """
    =================================================================
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    =================================================================
    ref: https://doi.org/10.1002/jcc.20237
    =================================================================
    """
    N = lambda x: torch.nn.functional.normalize(x, dim=-1)
    bc = N(b-c)
    n = N(torch.cross(N(b-a), bc))
    m = [bc, torch.cross(n, bc), n]
    d = [L * torch.cos(A), L * torch.sin(A) * torch.cos(D), -L * torch.sin(A) * torch.sin(D)]
    return c + sum([m[i] * d[i] for i in range(3)])


def residue_extend(phi, psi, omg, c0, n1, a1, add_ob=True):
    """
    ===============================================================
    input:   c(i-1), n(i),   ca(i)
    output:  c(i),   n(i+1), ca(i+1)
    ===============================================================
                     [b1]
                      |
   (c0-n1-a1) ->  c0-[n1-a1-c1]-n2-a2 -> (c1-n2-a2)
                         |
                        [o1]
    ===============================================================
    """
    c1 = extend(c0, n1, a1, torch.tensor(1.523, dtype=torch.float32), torch.tensor(1.941, dtype=torch.float32), phi)
    n2 = extend(n1, a1, c1, torch.tensor(1.329, dtype=torch.float32), torch.tensor(2.028, dtype=torch.float32), psi)
    a2 = extend(a1, c1, n2, torch.tensor(1.458, dtype=torch.float32), torch.tensor(2.124, dtype=torch.float32), omg)
    coord = [n1, a1, c1]
    if add_ob:
        b = extend(c1, n1, a1, torch.tensor(1.522, dtype=torch.float32), torch.tensor(1.927, dtype=torch.float32), torch.tensor(-2.143, dtype=torch.float32))
        o = extend(n2, a1, c1, torch.tensor(1.231, dtype=torch.float32), torch.tensor(2.108, dtype=torch.float32), torch.tensor(-3.142, dtype=torch.float32))
        coord += [b, o]
    return c1, n2, a2, coord


def dih_to_coord(dih, add_ob=True, return_raw=False, requires_grad=True):
    """
    =================================================================
    input:  (batch, length, dihedrals=(phi,psi,omega))
    output: (batch, length, atoms=(N,CA,C,CB,O), coords=(x,y,z))
    =================================================================
    """
    # transpose (batch,len,dih) -> (len,dih,batch,1)
    dih = dih.permute(1, 2, 0).unsqueeze(-1)
    batch_size = dih.shape[2]
    length = dih.shape[0]

    cn = torch.tensor(1.329, dtype=torch.float32)
    na = torch.tensor(1.458, dtype=torch.float32)
    cna = torch.tensor(2.124, dtype=torch.float32)
    ini_coords = [
        torch.tensor([0, 0, 0], dtype=torch.float32).to(dih.device).requires_grad_(requires_grad),
        torch.tensor([cn, 0, 0], dtype=torch.float32).to(dih.device).requires_grad_(requires_grad),
        torch.tensor([cn - na * torch.cos(cna), na * torch.sin(cna), 0], dtype=torch.float32).to(dih.device).requires_grad_(requires_grad)
    ]
    ini_coords = [ini.expand(batch_size, 3) for ini in ini_coords]

    coords = []
    c0, n1, a1 = ini_coords
    for dh in dih:
        c1, n2, a2, coord = residue_extend(dh[0], dh[1], dh[2], c0, n1, a1, add_ob=add_ob)
        c0, n1, a1 = c1, n2, a2
        coords.append(torch.stack(coord))

    coords = torch.stack(coords)

    if return_raw:
        return c0, n1, a1, coords
    else:
        coords = coords.permute(2, 0, 1, 3)
        return coords


def idealize(coords, gap_unmask, opt_iter=5000, add_ob=True, lr=0.001):
    '''
    ===================================================================
    input:  (batch, length, atoms=(N,CA,C), coords=(x,y,z))
    
    output: (batch, length, dihedrals=(phi,psi,omega))
            (batch, length, atoms=(N,CA,C), coords=(x,y,z))
    ===================================================================
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gap_unmask is None:
        gap_unmask = torch.ones(coords.shape[:2], dtype=torch.bool)
    else:
        assert gap_unmask.shape == coords.shape[:2]

    coords = coords.clone().to(device).detach().requires_grad_(True)
    init_coords = coords.clone().to(device).detach().requires_grad_(False)
    gap_unmask = gap_unmask.to(device).detach().requires_grad_(False)

    # Mask to ignore NaN values in the loss computation
    mask = ~(coords == 0).any(dim=-1).any(dim=-1)
    optimizer = torch.optim.Adam([coords], lr=lr)
    for k in range(opt_iter):
        optimizer.zero_grad()

        n, a, c = coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]
        # Bond lengths
        n_a = (((1.458 - to_len(n, a)[mask])**2)).mean()
        a_c = (((1.523 - to_len(a, c)[mask])**2)).mean()
        c_n = ((((1.329 - to_len(c[:, :-1], n[:, 1:]))[gap_unmask[:, 1:]])**2)).mean()

        # Bond angles
        n_a_c = (((1.941 - to_ang(n, a, c)[mask])**2)).mean()
        a_c_n = (((2.028 - to_ang(a[:, :-1], c[:, :-1], n[:, 1:])[gap_unmask[:, 1:]])**2)).mean()
        c_n_a = (((2.124 - to_ang(c[:, :-1], n[:, 1:], a[:, 1:])[gap_unmask[:, 1:]])**2)).mean()

        # Setup idealization loss function
        loss_ideal = n_a + a_c + c_n + n_a_c + a_c_n + c_n_a

        # RMS to initial coordinates
        loss_ms = ((coords[mask] - init_coords[mask])**2).mean()
        loss = loss_ideal + loss_ms
        loss.backward()
        optimizer.step()

        # # Debug information
        # if k % 50 == 0:
        #     print(f"Iteration {k+1}")
        #     print(f"loss_ideal: {loss_ideal.item()}, loss_ms: {loss_ms.item()}")
        #     print(f"coords grad norm: {coords.grad.norm().item()}")
        #     ideal_xyz = dih_to_coord(coord_to_dih(coords), add_ob=add_ob)
        #     dm_x = torch.sqrt(torch.sum((init_coords[:, None, :, :3] - init_coords[:, :, None, :3]) ** 2, dim=-1) + 1e-8)
        #     dm_y = torch.sqrt(torch.sum((ideal_xyz[:, None, :, :3] - ideal_xyz[:, :, None, :3]) ** 2, dim=-1) + 1e-8)
        #     rms = torch.sqrt(torch.mean((dm_x - dm_y) ** 2, dim=(1,2,3)))
        #     print(f"Iteration {k+1}, RMS: {rms.item()}")

    ideal_dih = coord_to_dih(coords)
    ideal_xyz = dih_to_coord(ideal_dih, add_ob=add_ob)
    return ideal_dih.detach().cpu().numpy(), ideal_xyz.detach().cpu().numpy()


def extract_backbone_coords_with_gaps(pdb_fname, backbone_atoms=['N','CA','C','CB','O'], chain_id=None):
    """
    Output the coordinates of the backbone atoms and the gaps between residues with
    shape: (residues, atoms, coords=(x,y,z))
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_fname)
    chain_id = os.path.basename(pdb_fname)[4] if chain_id is None else chain_id

    coords = []
    gaps = []
    residues = []
    resi_ids = []
    model = structure[0] # Taking the first model; alternatively skip this pdb file
    for chain in model:
        if chain.get_id() == chain_id or chain_id == "0":
            prev_residue_id = None
            for residue in chain:
                if all([atom in residue for atom in backbone_atoms]):
                    atoms_coord = [residue[atom].get_coord() for atom in backbone_atoms]
                    coords.append(atoms_coord)
                    residues.append(residue.get_resname())
                    current_residue_id = residue.get_id()[1]
                    resi_ids.append(str(current_residue_id))
                    current_residue_id = residue_id_to_ordinal(current_residue_id) if isinstance(current_residue_id, str) else current_residue_id
                    if prev_residue_id is None:
                        gaps.append(True)  # First residue, no gap before it
                    else:
                        gaps.append(current_residue_id == prev_residue_id + 1 or current_residue_id == prev_residue_id) # <= to handle 1A, 1, 2, 3 cases
                        # # Debug print statements to inspect gaps
                        # if chain_gaps[-1] == False:
                        #     print("Current id:", residue.get_id()[1])
                        #     print(f"Gap detected between {prev_residue_id} and {current_residue_id}")
                    prev_residue_id = current_residue_id

    coords_tensor = torch.from_numpy(np.array(coords))
    gaps_tensor = torch.tensor(gaps, dtype=torch.bool)

    return coords_tensor, gaps_tensor, residues, chain_id, resi_ids
    

def load_PDB(x, ln=None, atoms=["N", "CA", "C", "CB", "O"]):
    '''
    ===================================================================
    input:  x = PDB filename
            ln = length (optional)
            atoms = atoms to extract (optional)
            
    output: (batch, length, atoms, coords=(x,y,z))
    ===================================================================
    '''
    n, xyz, models = {}, {}, []

    def model_append():
        if len(n) > 0 and max(n.values()) > 0:
            max_length = max(n.values())
            for atom in atoms:
                while n[atom] < max_length:
                    xyz[atom].append([np.nan, np.nan, np.nan])
                    n[atom] += 1
            model = [xyz[atom] for atom in atoms]
            models.append(model)
            if ln is not None:
                for atom in atoms:
                    while n[atom] < ln:
                        xyz[atom].append([np.nan, np.nan, np.nan])
                        n[atom] += 1
        for atom in atoms:
            n[atom], xyz[atom] = 0, []

    model_append()

    for line in open(x, "r"):
        line = line.rstrip()
        if line[:5] == "MODEL":
            model_append()
        if line[:4] == "ATOM":
            atom = line[12:16].strip()
            resi = line[17:20]
            resn = int(line[22:26]) - 1
            x_coord = float(line[30:38])
            y_coord = float(line[38:46])
            z_coord = float(line[46:54])
            if atom in atoms:
                while n[atom] < resn:
                    xyz[atom].append([np.nan, np.nan, np.nan])
                    n[atom] += 1
                if n[atom] == resn:
                    xyz[atom].append([x_coord, y_coord, z_coord])
                    n[atom] += 1

    model_append()

    # Debug print statements to inspect shapes and content
    for i, model in enumerate(models):
        print(f"Model {i} lengths: {[len(model[atom]) for atom in range(len(atoms))]}")
    print(f"Final models shape: {[len(model) for model in models]}")

    # Convert models list to numpy array
    np_models = np.array(models)
    print(f"np_models shape before transpose: {np_models.shape}")

    return np_models.transpose(0, 2, 1, 3)


def save_PDB(coords, residues, chain_id, resi_ids, pdb_out, atoms=["N", "CA", "C", "CB", "O"]):
    '''
    ===================================================================
    input: (batch, length, atoms=(N, CA, C, CB, O), coords=(x, y, z))
    ===================================================================
    '''
    num_models = coords.shape[0]
    out = open(pdb_out, "w")
    k = 1
    for m, model in enumerate(coords):
        if num_models > 1:
            out.write("MODEL    %5d\n" % (m + 1))
        for r, residue in enumerate(model[:len(residues)]):
            res_name = residues[r]
            res_id = resi_ids[r]
            for a, atom in enumerate(residue):
                x, y, z = atom
                if not np.isnan(x):
                    out.write(
                        "ATOM  %5d  %-2s  %3s %s%4s    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                        % (k, atoms[a], res_name, chain_id, res_id, x, y, z, 1.00, 0.00)
                    )
                k += 1
        if num_models > 1:
            out.write("ENDMDL\n")
    out.close()


def main():
    """On the fly testing"""
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    source = PDBFile.read(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/1CRN.pdb")
    )
    source_struct = source.get_structure()
    # print(source_struct[0])
    phi, psi, omega = [torch.tensor(x) for x in struc.dihedral_backbone(source_struct)]

    builder = NERFBuilder(phi, psi, omega)
    print(builder.cartesian_coords)
    print(builder.cartesian_coords.shape)


if __name__ == "__main__":
    main()
