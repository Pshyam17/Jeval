import numpy as np


class UMAPProjector:
    def __init__(
        self,
        n_components: int = 3,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ):
        self._reducer = None
        self._fitted = False
        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._min_dist = min_dist
        self._buffer: list[np.ndarray] = []
        self._min_fit_size = 10
        # per-dimension ranges captured at fit time, used for all future transforms
        self._coord_min: np.ndarray | None = None
        self._coord_span: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> None:
        import umap
        self._reducer = umap.UMAP(
            n_components=self._n_components,
            n_neighbors=min(self._n_neighbors, len(embeddings) - 1),
            min_dist=self._min_dist,
            random_state=42,
        )
        raw = self._reducer.fit_transform(embeddings)
        # capture per-dimension ranges for stable future scaling
        self._coord_min = raw.min(axis=0)
        self._coord_span = raw.max(axis=0) - raw.min(axis=0)
        self._fitted = True

    def transform(self, embedding: np.ndarray) -> list[float]:
        if not self._fitted:
            self._buffer.append(embedding)
            if len(self._buffer) >= self._min_fit_size:
                self.fit(np.array(self._buffer))
                # re-project all buffered points with the stable normalisation
                coords = self._reducer.transform(np.array(self._buffer))
                normalised = self._normalise(coords)
                return normalised[-1].tolist()
            return [0.0, 0.0, 0.0]
        coords = self._reducer.transform(embedding.reshape(1, -1))
        return self._normalise(coords)[0].tolist()

    def _normalise(self, coords: np.ndarray) -> np.ndarray:
        coords = coords.copy().astype(float)
        for dim in range(coords.shape[1]):
            span = self._coord_span[dim] if self._coord_span is not None else 0.0
            mn = self._coord_min[dim] if self._coord_min is not None else 0.0
            if span > 0:
                coords[:, dim] = (coords[:, dim] - mn) / span * 10 - 5
            else:
                coords[:, dim] = 0.0
        # clamp: new points can project beyond the training range
        return np.clip(coords, -5.0, 5.0)

    def seed_from_encoder(self, encoder, texts: list[str]) -> None:
        embeddings = encoder.encode(texts)
        self.fit(embeddings)
