"""
Schema induction for v3.0 architecture.

Implements offline schema discovery from novel artifacts (§1.11).
Not used during live episodes - runs as batch processing.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StructuralSignature:
    """Structural signature of an artifact for clustering."""
    field_names: frozenset[str] = field(default_factory=frozenset)
    value_types: frozenset[str] = field(default_factory=frozenset)
    nesting_depth: int = 0
    key_patterns: frozenset[str] = field(default_factory=frozenset)


@dataclass
class InducedSchema:
    """A schema induced from a cluster of similar artifacts."""
    name: str
    required_fields: dict[str, str]  # field_name -> regex pattern
    optional_fields: dict[str, str]  # field_name -> regex pattern
    cluster_size: int
    structural_signature: StructuralSignature


class SchemaInduction:
    """
    Offline schema discovery from novel artifacts (§1.11).

    Process:
    1. Cluster unknown artifacts by structural signature
    2. If cluster size >= m_min, extract common fields
    3. Identify required fields (present in >= 80% of members)
    4. Identify optional fields (present in 20-80% of members)
    5. Create new schema and add to registry
    """

    def __init__(
        self,
        clustering_threshold: float = 0.30,  # δ: Jaccard distance threshold
        min_cluster_size: int = 5,  # m_min: minimum cluster size
        required_threshold: float = 0.80,  # 80% presence for required fields
        optional_threshold: float = 0.20,  # 20% presence for optional fields
    ):
        self._clustering_threshold = clustering_threshold
        self._min_cluster_size = min_cluster_size
        self._required_threshold = required_threshold
        self._optional_threshold = optional_threshold

    def compute_signature(self, artifact: str) -> StructuralSignature:
        """
        Extract structural signature from artifact.

        Signature components:
        - field_names: keys found in JSON-like or key:value patterns
        - value_types: detected value types (string, number, boolean, null)
        - nesting_depth: maximum nesting level
        - key_patterns: common key naming patterns (snake_case, camelCase, etc.)
        """
        field_names: set[str] = set()
        value_types: set[str] = set()
        max_depth = 0
        key_patterns: set[str] = set()

        # Detect JSON-like structures
        json_key_pattern = re.compile(r'"([^"]+)"\s*:')
        for match in json_key_pattern.finditer(artifact):
            key = match.group(1)
            field_names.add(key)
            key_patterns.add(self._classify_key_pattern(key))

        # Detect key:value patterns (non-JSON)
        kv_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*([^,\n}]+)')
        for match in kv_pattern.finditer(artifact):
            key = match.group(1)
            value = match.group(2)
            field_names.add(key)
            key_patterns.add(self._classify_key_pattern(key))
            value_types.add(self._detect_value_type(value))

        # Detect value types from context
        if re.search(r'\b(true|false)\b', artifact):
            value_types.add('boolean')
        if re.search(r'\bnull\b', artifact):
            value_types.add('null')
        if re.search(r'\b\d+\b', artifact):
            value_types.add('number')
        if re.search(r'"[^"]+"', artifact):
            value_types.add('string')

        # Compute nesting depth
        max_depth = self._compute_nesting_depth(artifact)

        return StructuralSignature(
            field_names=frozenset(field_names),
            value_types=frozenset(value_types),
            nesting_depth=max_depth,
            key_patterns=frozenset(key_patterns),
        )

    def _classify_key_pattern(self, key: str) -> str:
        """Classify key naming pattern."""
        if '_' in key and key.islower():
            return 'snake_case'
        if re.match(r'^[a-z][a-zA-Z0-9]*$', key):
            return 'camelCase'
        if re.match(r'^[A-Z][a-zA-Z0-9]*$', key):
            return 'PascalCase'
        if re.match(r'^[A-Z][A-Z0-9_]*$', key):
            return 'SCREAMING_SNAKE'
        return 'mixed'

    def _detect_value_type(self, value: str) -> str:
        """Detect value type from string representation."""
        value = value.strip().strip('"\'')
        if value.lower() in ('true', 'false'):
            return 'boolean'
        if value.lower() == 'null':
            return 'null'
        if re.match(r'^-?\d+$', value):
            return 'integer'
        if re.match(r'^-?\d+\.\d+$', value):
            return 'float'
        return 'string'

    def _compute_nesting_depth(self, artifact: str) -> int:
        """Compute maximum nesting depth."""
        max_depth = 0
        current_depth = 0
        for char in artifact:
            if char in '{[':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in '}]':
                current_depth = max(0, current_depth - 1)
        return max_depth

    def jaccard_distance(self, sig1: StructuralSignature, sig2: StructuralSignature) -> float:
        """
        Compute Jaccard distance between two structural signatures.

        distance = 1 - |A ∩ B| / |A ∪ B|

        Primary comparison is on field_names (most discriminative).
        """
        set1 = sig1.field_names
        set2 = sig2.field_names

        if not set1 and not set2:
            return 0.0
        if not set1 or not set2:
            return 1.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return 1.0 - (intersection / union)

    def cluster_artifacts(
        self,
        artifacts: list[str],
    ) -> list[list[int]]:
        """
        Cluster artifacts by structural signature using hierarchical clustering.

        Args:
            artifacts: List of artifact texts

        Returns:
            List of clusters, where each cluster is a list of artifact indices
        """
        n = len(artifacts)
        if n == 0:
            return []

        # Compute signatures for all artifacts
        signatures = [self.compute_signature(art) for art in artifacts]

        # Hierarchical clustering with single linkage
        clusters: list[list[int]] = [[i] for i in range(n)]
        cluster_sigs = [signatures[i] for i in range(n)]

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(clusters):
                j = i + 1
                while j < len(clusters):
                    # Compute min distance between clusters (single linkage)
                    min_dist = 1.0
                    for idx1 in clusters[i]:
                        for idx2 in clusters[j]:
                            dist = self.jaccard_distance(signatures[idx1], signatures[idx2])
                            if dist < min_dist:
                                min_dist = dist

                    # Merge if below threshold
                    if min_dist <= self._clustering_threshold:
                        clusters[i].extend(clusters.pop(j))
                        changed = True
                    else:
                        j += 1
                i += 1

        # Filter to clusters that meet minimum size
        return [c for c in clusters if len(c) >= self._min_cluster_size]

    def induce_schema(
        self,
        artifacts: list[str],
        cluster_indices: list[int],
        schema_name: Optional[str] = None,
    ) -> Optional[InducedSchema]:
        """
        Induce a schema from a cluster of artifacts.

        Args:
            artifacts: Full list of artifacts
            cluster_indices: Indices of artifacts in this cluster
            schema_name: Optional name for the induced schema

        Returns:
            InducedSchema if successful, None if cluster too small
        """
        if len(cluster_indices) < self._min_cluster_size:
            return None

        cluster_artifacts = [artifacts[i] for i in cluster_indices]

        # Extract all fields from each artifact
        artifact_fields: list[dict[str, str]] = []
        for art in cluster_artifacts:
            fields = self._extract_fields_with_patterns(art)
            artifact_fields.append(fields)

        # Aggregate field presence
        field_counts: Counter = Counter()
        field_patterns: defaultdict[str, list[str]] = defaultdict(list)

        for fields in artifact_fields:
            for field_name, pattern_match in fields.items():
                field_counts[field_name] += 1
                field_patterns[field_name].append(pattern_match)

        # Classify fields as required or optional
        required_fields: dict[str, str] = {}
        optional_fields: dict[str, str] = {}

        n_cluster = len(cluster_artifacts)
        for field_name, count in field_counts.items():
            presence_ratio = count / n_cluster
            # Derive representative pattern
            patterns = field_patterns[field_name]
            representative_pattern = self._derive_representative_pattern(patterns)

            if presence_ratio >= self._required_threshold:
                required_fields[field_name] = representative_pattern
            elif presence_ratio >= self._optional_threshold:
                optional_fields[field_name] = representative_pattern

        # Compute average structural signature for cluster
        signatures = [self.compute_signature(art) for art in cluster_artifacts]
        avg_signature = StructuralSignature(
            field_names=frozenset(signatures[0].field_names),  # Use first as representative
            value_types=frozenset().union(*[s.value_types for s in signatures]),
            nesting_depth=int(sum(s.nesting_depth for s in signatures) / len(signatures)),
            key_patterns=frozenset().union(*[s.key_patterns for s in signatures]),
        )

        return InducedSchema(
            name=schema_name or f"induced_{len(required_fields)}_{len(optional_fields)}",
            required_fields=required_fields,
            optional_fields=optional_fields,
            cluster_size=n_cluster,
            structural_signature=avg_signature,
        )

    def _extract_fields_with_patterns(self, artifact: str) -> dict[str, str]:
        """Extract field names and their matched patterns from artifact."""
        fields: dict[str, str] = {}

        # JSON-like keys
        json_pattern = re.compile(r'"([^"]+)"\s*:\s*"([^"]+)"')
        for match in json_pattern.finditer(artifact):
            key, value = match.groups()
            fields[key] = re.escape(value)

        # Key: value patterns
        kv_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*([^,\n}]+)')
        for match in kv_pattern.finditer(artifact):
            key, value = match.groups()
            fields[key] = self._value_to_pattern(value.strip())

        return fields

    def _value_to_pattern(self, value: str) -> str:
        """Convert a value string to a regex pattern."""
        if value.isdigit():
            return r'\d+'
        if re.match(r'^\d+\.\d+$', value):
            return r'\d+\.\d+'
        if value.lower() in ('true', 'false'):
            return r'(true|false)'
        return re.escape(value)

    def _derive_representative_pattern(self, patterns: list[str]) -> str:
        """Derive a representative regex pattern from a list of matched values."""
        if not patterns:
            return r'.*'

        # Check if all are numeric
        if all(p.isdigit() or re.match(r'^\d+$', p) for p in patterns):
            return r'\d+'

        # Check if all are boolean
        if all(p.lower() in ('true', 'false') for p in patterns):
            return r'(true|false)'

        # Check if all match a common pattern
        if all(re.match(r'^[\w./\-]+\.\w{2,4}$', p) for p in patterns):
            return r'[\w./\-]+\.\w{2,4}'  # File paths

        # Default: match any of the observed values
        if len(set(patterns)) <= 3:
            return '(' + '|'.join(re.escape(p) for p in set(patterns)) + ')'

        return r'.+'
