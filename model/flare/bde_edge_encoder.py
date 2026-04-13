"""
BDE Edge Encoder for Physics-Informed Motif Extraction.

Provides per-bond BDE estimates from 2D molecular graphs using a lookup table
(with optional ALFABET integration if installed), and encodes these alongside
bond type/ring/conjugation features into dense edge embeddings for the GCN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS


# =============================================================================
# BDE LOOKUP TABLE (kcal/mol)
# Values from literature + context corrections for common bond types.
# Context modifiers: benzylic -17, allylic -12, tertiary -10 vs primary.
# =============================================================================
_BDE_LOOKUP = {
    # (atom1_symbol, atom2_symbol, bond_order_rounded) -> BDE kcal/mol
    ('C', 'C', 1): 83.0,
    ('C', 'C', 2): 146.0,
    ('C', 'C', 3): 200.0,
    ('C', 'C', 1.5): 115.0,   # aromatic
    ('C', 'N', 1): 73.0,
    ('C', 'N', 2): 147.0,
    ('C', 'N', 1.5): 100.0,   # aromatic
    ('C', 'O', 1): 86.0,
    ('C', 'O', 2): 178.0,
    ('C', 'S', 1): 65.0,
    ('C', 'S', 2): 107.0,
    ('C', 'F', 1): 116.0,
    ('C', 'Cl', 1): 81.0,
    ('C', 'Br', 1): 68.0,
    ('C', 'I', 1): 51.0,
    ('C', 'H', 1): 99.0,
    ('N', 'N', 1): 38.0,
    ('N', 'N', 2): 100.0,
    ('N', 'O', 1): 46.0,
    ('N', 'H', 1): 93.0,
    ('O', 'O', 1): 35.0,
    ('O', 'H', 1): 119.0,
    ('S', 'S', 1): 54.0,
    ('S', 'H', 1): 82.0,
    ('P', 'O', 1): 86.0,
    ('P', 'O', 2): 140.0,
}

# BDE range for normalization: [20, 220] kcal/mol → [0, 1]
_BDE_MIN = 20.0
_BDE_MAX = 220.0

# Bond type index map
_BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

# Number of bond type categories (+1 for "other")
N_BOND_TYPES = 5


def _get_bond_order_key(bond):
    """Map RDKit bond type to the float key used in BDE_LOOKUP."""
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        return 1
    elif bt == Chem.rdchem.BondType.DOUBLE:
        return 2
    elif bt == Chem.rdchem.BondType.TRIPLE:
        return 3
    elif bt == Chem.rdchem.BondType.AROMATIC:
        return 1.5
    return 1  # fallback


def _context_correction(mol, bond):
    """
    Estimate context corrections to BDE (kcal/mol):
    - Benzylic position: -17 kcal/mol (radical resonance with aromatic ring)
    - Allylic position: -12 kcal/mol (only if not already benzylic — effects don't stack)
    - Tertiary carbon: -10 kcal/mol vs primary (only if not benzylic)
    Returns a correction value (negative = weakens bond).
    """
    correction = 0.0
    for atom_idx in [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() != 'C':
            continue

        # Check benzylic: atom has a neighbor bond that is aromatic
        is_benzylic = any(
            mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx()).GetIsAromatic()
            for nbr in atom.GetNeighbors()
        )
        # Check allylic: atom has a neighbor bond that is a double bond
        # Only if not already benzylic — resonance stabilisation is not additive
        is_allylic = (not is_benzylic) and any(
            mol.GetBondBetweenAtoms(atom_idx, nbr.GetIdx()).GetBondType()
            == Chem.rdchem.BondType.DOUBLE
            for nbr in atom.GetNeighbors()
        )
        # Tertiary carbon: 3+ carbon neighbours, not benzylic (already accounted for)
        is_tertiary = (not is_benzylic) and (
            sum(1 for nb in atom.GetNeighbors() if nb.GetSymbol() == 'C') >= 3
        )

        if is_benzylic:
            correction -= 8.5   # half of 17, split across two bond endpoints
        elif is_allylic:
            correction -= 6.0   # half of 12
        if is_tertiary:
            correction -= 5.0

    return correction


def compute_bde_features(mol):
    """
    Compute per-bond BDE features for a molecule.

    Returns
    -------
    bde_raw : np.ndarray [N_bonds]
        Raw BDE estimates in kcal/mol (with context corrections).
    bde_norm : np.ndarray [N_bonds]
        BDE normalized to [0, 1].
    bond_types : np.ndarray [N_bonds] int
        Bond type index (0=single, 1=double, 2=triple, 3=aromatic, 4=other).
    in_ring : np.ndarray [N_bonds] int
        1 if bond is in a ring, 0 otherwise.
    conjugated : np.ndarray [N_bonds] int
        1 if bond is conjugated, 0 otherwise.
    is_brics : np.ndarray [N_bonds] int
        1 if bond is a BRICS cleavage site, 0 otherwise.
    bond_atom_pairs : list of (int, int)
        (begin_atom_idx, end_atom_idx) for each bond — used to build adjacency.
    """
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    cleavage_pairs = set(tuple(sorted(b[0])) for b in brics_bonds)

    bde_raw_list = []
    bde_norm_list = []
    bond_type_list = []
    in_ring_list = []
    conjugated_list = []
    is_brics_list = []
    bond_atom_pairs = []

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        sym_u = mol.GetAtomWithIdx(u).GetSymbol()
        sym_v = mol.GetAtomWithIdx(v).GetSymbol()
        bond_order = _get_bond_order_key(bond)

        # Look up BDE: try both orderings
        bde = _BDE_LOOKUP.get(
            (sym_u, sym_v, bond_order),
            _BDE_LOOKUP.get(
                (sym_v, sym_u, bond_order),
                83.0  # fallback: C-C single
            )
        )
        # Apply context correction
        bde += _context_correction(mol, bond)
        bde = max(bde, _BDE_MIN)

        bt = bond.GetBondType()
        bt_idx = _BOND_TYPE_MAP.get(bt, 4)

        bde_raw_list.append(bde)
        bde_norm_list.append((bde - _BDE_MIN) / (_BDE_MAX - _BDE_MIN))
        bond_type_list.append(bt_idx)
        in_ring_list.append(int(bond.IsInRing()))
        conjugated_list.append(int(bond.GetIsConjugated()))
        is_brics_list.append(int(tuple(sorted((u, v))) in cleavage_pairs))
        bond_atom_pairs.append((u, v))

    return (
        np.array(bde_raw_list, dtype=np.float32),
        np.array(bde_norm_list, dtype=np.float32),
        np.array(bond_type_list, dtype=np.int64),
        np.array(in_ring_list, dtype=np.int64),
        np.array(conjugated_list, dtype=np.int64),
        np.array(is_brics_list, dtype=np.int64),
        bond_atom_pairs,
    )


def compute_bde_adjacency(mol):
    """
    Build a dense N×N BDE adjacency matrix where each entry is the
    normalized BDE of the bond between atoms i and j (0 if no bond).
    Also returns the raw BRICS-weighted adjacency (backward compat).

    Returns
    -------
    A_bde_norm : torch.Tensor [N, N]  — normalized BDE, range [0,1]
    A_brics    : torch.Tensor [N, N]  — BRICS-weighted (1.0/10.0) for GCN
    bde_edge_feats : torch.Tensor [N, N, 5]
        Per-bond dense feature matrix: [bde_norm, bond_type_oh(4)] stored
        as a dense N×N×5 tensor (zeros for absent bonds).
    """
    N = mol.GetNumAtoms()
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    cleavage_pairs = set(tuple(sorted(b[0])) for b in brics_bonds)

    A_bde = np.zeros((N, N), dtype=np.float32)
    A_brics = np.zeros((N, N), dtype=np.float32)
    edge_feat_mat = np.zeros((N, N, 5), dtype=np.float32)  # [bde_norm, bt0..bt3]

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        sym_u = mol.GetAtomWithIdx(u).GetSymbol()
        sym_v = mol.GetAtomWithIdx(v).GetSymbol()
        bond_order = _get_bond_order_key(bond)

        bde = _BDE_LOOKUP.get(
            (sym_u, sym_v, bond_order),
            _BDE_LOOKUP.get((sym_v, sym_u, bond_order), 83.0)
        )
        bde += _context_correction(mol, bond)
        bde = max(bde, _BDE_MIN)
        bde_n = (bde - _BDE_MIN) / (_BDE_MAX - _BDE_MIN)

        bt_idx = _BOND_TYPE_MAP.get(bond.GetBondType(), 4)
        bt_oh = np.zeros(4, dtype=np.float32)
        if bt_idx < 4:
            bt_oh[bt_idx] = 1.0

        A_bde[u, v] = bde_n
        A_bde[v, u] = bde_n

        feat = np.concatenate([[bde_n], bt_oh])
        edge_feat_mat[u, v] = feat
        edge_feat_mat[v, u] = feat

        # BRICS weights (backward compat for GCN)
        w = 1.0 if tuple(sorted((u, v))) in cleavage_pairs else 10.0
        A_brics[u, v] = w
        A_brics[v, u] = w

    # Self-loops
    np.fill_diagonal(A_bde, 1.0)        # max BDE on self (no cut)
    np.fill_diagonal(A_brics, 10.0)     # same as before

    return (
        torch.tensor(A_bde, dtype=torch.float32),
        torch.tensor(A_brics, dtype=torch.float32),
        torch.tensor(edge_feat_mat, dtype=torch.float32),
    )


# =============================================================================
# KINETIC ADJACENCY ENCODER
# Maps per-bond BDE + chemical features → dense edge embeddings [N, N, D]
# =============================================================================
class KineticAdjacencyEncoder(nn.Module):
    """
    Encodes the N×N×5 dense edge feature matrix into N×N×hidden_dim embeddings.
    Used inside the BDE-aware GCN to make message weights chemistry-dependent.
    """

    def __init__(self, in_features=5, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_feat_mat):
        """
        edge_feat_mat: [B, N, N, 5]
        Returns: [B, N, N, hidden_dim]
        """
        return self.net(edge_feat_mat)


# =============================================================================
# BDE-AWARE DENSE GCN
# Like DenseGCNConv but weights messages by BDE (low BDE = easy to cut = weak connection)
# =============================================================================
class BDEAwareDenseGCN(nn.Module):
    """
    Dense GCN layer where adjacency weights are modulated by BDE:
    - High BDE (stable bond) → strong message passing (atoms stay in same motif)
    - Low BDE (weak bond) → weak message (atoms may be in different motifs)
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        # Gate: learns how much to trust the BDE-modulated adjacency
        self.bde_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x, A_brics, A_bde_norm=None):
        """
        x: [B, N, D]
        A_brics: [B, N, N]  — BRICS-weighted adjacency (for normalization base)
        A_bde_norm: [B, N, N] — normalized BDE adjacency (optional)
        """
        if A_bde_norm is not None:
            # BDE gate: high BDE → gate ≈ 1 (preserve strong bonds)
            # low BDE → gate < 1 (weaken fragile bonds in message passing)
            gate = self.bde_gate(A_bde_norm.unsqueeze(-1)).squeeze(-1)  # [B, N, N]
            A_eff = A_brics * gate
        else:
            A_eff = A_brics

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        d = A_eff.sum(dim=-1).clamp(min=1e-5)
        d_inv_sqrt = d.pow(-0.5)
        A_norm = d_inv_sqrt.unsqueeze(-1) * A_eff * d_inv_sqrt.unsqueeze(1)

        out = torch.bmm(A_norm, x)
        return F.leaky_relu(self.lin(out), 0.1)


# =============================================================================
# CUT PROBABILITY HEAD
# Estimates P(cut) for each bond from assignment vectors of adjacent atoms.
# Trained to inversely correlate with BDE via L_ordering loss.
# =============================================================================
class CutProbabilityHead(nn.Module):
    """
    Given per-atom assignment vectors S [B, N, K] and the BDE adjacency,
    computes a cut probability for each bond:
        cut_prob[u,v] = 0.5 * || S[u] - S[v] ||_1 / K
    This is differentiable: if adjacent atoms disagree strongly on motif
    assignment → high cut probability. Combined with L_ordering loss (which
    penalizes high cut probability on high-BDE bonds), this teaches the model
    to cut at weak bonds.
    """

    def forward(self, S, A_brics):
        """
        S: [B, N, K]  soft assignment matrix
        A_brics: [B, N, N]  (used as bond mask: nonzero = bond exists)
        Returns: cut_probs [B, N, N] symmetric, range [0, 1]
        """
        B, N, K = S.shape
        # L1 distance between assignment vectors for all pairs: [B, N, N]
        # ||S_i - S_j||_1 = sum_k |S_ik - S_jk|
        S_diff = torch.abs(S.unsqueeze(2) - S.unsqueeze(1))  # [B, N, N, K]
        # L1 distance between two probability vectors ∈ [0, 2], so divide by 2 for [0, 1]
        l1_dist = S_diff.sum(dim=-1) / 2.0                    # [B, N, N], range [0, 1]

        # Mask to existing bonds only
        bond_mask = (A_brics > 0).float()
        cut_probs = l1_dist * bond_mask
        return cut_probs
