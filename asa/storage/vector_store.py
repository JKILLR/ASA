"""
AtomicVectorStore - Storage for atomic embeddings with similarity search.

Provides:
- Add/get atomic embeddings
- Similarity search by shell
- Persistence (save/load)
"""

import json
from typing import Dict, List, Optional

import numpy as np
import torch

from ..core.config import AtomConfig


class AtomicVectorStore:
    """
    Storage for atomic embeddings with similarity search.

    Stores atomic data (nuclear, shells, charge, valence) and provides
    fast similarity search capabilities.
    """

    def __init__(self, config: AtomConfig):
        """
        Initialize vector store.

        Args:
            config: Atom configuration
        """
        self.config = config
        self.atoms: Dict[str, Dict] = {}
        self.token_to_id: Dict[str, int] = {}
        self._next_id = 0

    def add(self, token: str, atom_data: Dict[str, torch.Tensor]) -> int:
        """
        Add atom data to store.

        Args:
            token: Token string
            atom_data: Dict of atom components

        Returns:
            Assigned token ID
        """
        if token not in self.token_to_id:
            self.token_to_id[token] = self._next_id
            self._next_id += 1

        stored = {"token_id": self.token_to_id[token], "token": token}

        for k, v in atom_data.items():
            if isinstance(v, torch.Tensor):
                stored[k] = v.detach().cpu().numpy()
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                stored[k] = [t.detach().cpu().numpy() for t in v]
            else:
                stored[k] = v

        self.atoms[token] = stored
        return stored["token_id"]

    def get(self, token: str) -> Optional[Dict]:
        """
        Get atom data by token.

        Args:
            token: Token string

        Returns:
            Atom data dict with tensors, or None
        """
        stored = self.atoms.get(token)
        if not stored:
            return None

        result = {}
        for k, v in stored.items():
            if isinstance(v, np.ndarray):
                result[k] = torch.from_numpy(v)
            elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                result[k] = [torch.from_numpy(arr) for arr in v]
            else:
                result[k] = v

        return result

    def get_by_id(self, token_id: int) -> Optional[Dict]:
        """
        Get atom data by token ID.

        Args:
            token_id: Token ID

        Returns:
            Atom data dict, or None
        """
        for token, data in self.atoms.items():
            if data.get("token_id") == token_id:
                return self.get(token)
        return None

    def search_similar(
        self,
        query: Dict,
        shell: int = 1,
        k: int = 10,
        exclude_tokens: List[str] = None,
    ) -> List[tuple]:
        """
        Find similar atoms by shell vector.

        Args:
            query: Query atom data
            shell: Shell level to compare (1-indexed)
            k: Number of results
            exclude_tokens: Tokens to exclude from results

        Returns:
            List of (token, similarity) tuples
        """
        exclude_tokens = exclude_tokens or []

        # Get query vector
        if "shells" not in query:
            return []

        query_shells = query["shells"]
        if shell - 1 >= len(query_shells):
            return []

        query_vec = query_shells[shell - 1]
        if isinstance(query_vec, torch.Tensor):
            query_vec = query_vec.numpy()

        similarities = []
        for token, stored in self.atoms.items():
            if token in exclude_tokens:
                continue
            if "shells" not in stored:
                continue

            stored_shells = stored["shells"]
            if shell - 1 >= len(stored_shells):
                continue

            stored_vec = stored_shells[shell - 1]

            # Cosine similarity
            sim = np.dot(query_vec, stored_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8
            )
            similarities.append((token, float(sim)))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]

    def search_by_nuclear(
        self, query: Dict, k: int = 10
    ) -> List[tuple]:
        """
        Find similar atoms by nuclear vector.

        Args:
            query: Query atom data
            k: Number of results

        Returns:
            List of (token, similarity) tuples
        """
        if "nuclear" not in query:
            return []

        query_vec = query["nuclear"]
        if isinstance(query_vec, torch.Tensor):
            query_vec = query_vec.numpy()

        similarities = []
        for token, stored in self.atoms.items():
            if "nuclear" not in stored:
                continue

            stored_vec = stored["nuclear"]
            sim = np.dot(query_vec, stored_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-8
            )
            similarities.append((token, float(sim)))

        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]

    def search_by_charge(
        self,
        target_polarity: float,
        tolerance: float = 0.3,
    ) -> List[tuple]:
        """
        Find atoms by charge polarity.

        Args:
            target_polarity: Target polarity value
            tolerance: Allowed deviation

        Returns:
            List of (token, polarity) tuples
        """
        results = []
        for token, stored in self.atoms.items():
            polarity = stored.get("polarity")
            if polarity is not None:
                if isinstance(polarity, np.ndarray):
                    polarity = float(polarity.mean())
                if abs(polarity - target_polarity) <= tolerance:
                    results.append((token, polarity))

        results.sort(key=lambda x: abs(x[1] - target_polarity))
        return results

    def save(self, path: str) -> None:
        """
        Save store to file.

        Args:
            path: File path
        """
        serializable = {}
        for token, data in self.atoms.items():
            serializable[token] = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    serializable[token][k] = v.tolist()
                elif isinstance(v, list) and v and isinstance(v[0], np.ndarray):
                    serializable[token][k] = [arr.tolist() for arr in v]
                else:
                    serializable[token][k] = v

        with open(path, "w") as f:
            json.dump(
                {
                    "atoms": serializable,
                    "token_to_id": self.token_to_id,
                    "next_id": self._next_id,
                },
                f,
            )

    def load(self, path: str) -> None:
        """
        Load store from file.

        Args:
            path: File path
        """
        with open(path, "r") as f:
            data = json.load(f)

        self.token_to_id = data["token_to_id"]
        self._next_id = data.get("next_id", len(self.token_to_id))

        for token, atom_data in data["atoms"].items():
            restored = {}
            for k, v in atom_data.items():
                if isinstance(v, list):
                    if v and isinstance(v[0], list):
                        # List of arrays
                        restored[k] = [np.array(arr) for arr in v]
                    elif v and isinstance(v[0], (int, float)):
                        # Single array
                        restored[k] = np.array(v)
                    else:
                        restored[k] = v
                else:
                    restored[k] = v
            self.atoms[token] = restored

    def __len__(self) -> int:
        return len(self.atoms)

    def __contains__(self, token: str) -> bool:
        return token in self.atoms

    def tokens(self) -> List[str]:
        """Get all stored tokens."""
        return list(self.atoms.keys())

    def clear(self) -> None:
        """Clear all stored data."""
        self.atoms.clear()
        self.token_to_id.clear()
        self._next_id = 0
