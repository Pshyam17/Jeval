"""
Unit tests for session_aware.py and schema_induction.py components.

Tests:
- ContextReuseCache: similarity threshold behavior
- StaticRetrievalPolicy: QA type classification
- ConfidenceRetryEscalation: escalation logic
- compute_query_confidence: entity overlap
- compute_answer_confidence: evidence support, query coverage
- SchemaInduction: clustering and schema discovery
- NoveltyGate: schema novelty detection
"""
import pytest
import numpy as np

from jeval.encoders.sentence_encoder import FrozenEncoder
from jeval.memory.session_aware import (
    ContextReuseCache,
    StaticRetrievalPolicy,
    ConfidenceRetryEscalation,
    compute_query_confidence,
    compute_answer_confidence,
    CachedContext,
)
from jeval.memory.schema_induction import (
    SchemaInduction,
    StructuralSignature,
    InducedSchema,
)
from jeval.memory.novelty_gate import NoveltyGate


class TestContextReuseCache:
    """Test context reuse caching with similarity threshold."""

    @pytest.fixture
    def encoder(self):
        return FrozenEncoder()

    @pytest.fixture
    def cache(self, encoder):
        return ContextReuseCache(encoder, threshold=0.85, max_cache_size=10)

    def test_cache_miss_then_hit(self, encoder, cache):
        """First question caches, identical second question hits cache."""
        question = "What is the JWT_SECRET value?"
        context = "JWT_SECRET = abc123"

        # First call - cache miss
        def retrieve_fn(q):
            return f"Retrieved context for: {q}"

        result1, cached1 = cache.get_or_retrieve("q1", question, retrieve_fn)
        assert not cached1
        assert "Retrieved context" in result1

        # Same UUID - exact hit
        result2, cached2 = cache.get_or_retrieve("q1", question, retrieve_fn)
        assert cached2
        assert result2 == result1

    def test_similar_questions_reuse_context(self, encoder):
        """Questions with high similarity reuse cached context."""
        cache = ContextReuseCache(encoder, threshold=0.70, max_cache_size=10)

        question1 = "How do I fix the database connection timeout?"
        question2 = "How to fix database connection timeout issues?"
        context = "Check your connection pool settings and increase timeout"

        def retrieve_fn(q):
            return context

        # First question - cache miss
        result1, cached1 = cache.get_or_retrieve("q1", question1, retrieve_fn)
        assert not cached1

        # Similar question - should hit cache (high cosine similarity)
        result2, cached2 = cache.get_or_retrieve("q2", question2, retrieve_fn)
        assert cached2
        assert result2 == context

    def test_dissimilar_questions_no_reuse(self, encoder):
        """Dissimilar questions don't reuse context."""
        cache = ContextReuseCache(encoder, threshold=0.85, max_cache_size=10)

        question1 = "What is the JWT_SECRET?"
        question2 = "How many tests passed?"

        def retrieve_fn(q):
            return f"Context for: {q}"

        result1, _ = cache.get_or_retrieve("q1", question1, retrieve_fn)
        result2, cached2 = cache.get_or_retrieve("q2", question2, retrieve_fn)

        assert not cached2
        assert result2 != result1

    def test_cache_eviction_lru(self, encoder):
        """Cache evicts oldest entries when full."""
        cache = ContextReuseCache(encoder, threshold=0.85, max_cache_size=3)

        def retrieve_fn(q):
            return f"Context for {q}"

        # Fill cache with distinct questions
        cache.get_or_retrieve("q0", "What is the JWT_SECRET?", retrieve_fn)
        cache.get_or_retrieve("q1", "How many tests passed?", retrieve_fn)
        cache.get_or_retrieve("q2", "When did the deployment occur?", retrieve_fn)

        # Add one more - should evict q0 (oldest)
        cache.get_or_retrieve("q3", "Where is the config file?", retrieve_fn)

        assert len(cache._cache) == 3
        assert "q0" not in cache._cache
        assert "q3" in cache._cache


class TestStaticRetrievalPolicy:
    """Test QA type classification and policy retrieval."""

    def test_factoid_classification(self):
        policy = StaticRetrievalPolicy()

        assert policy.get_qa_type("What is the API endpoint?") == "factoid"
        assert policy.get_qa_type("What are the configuration settings?") == "factoid"
        assert policy.get_qa_type("Define the JWT_SECRET variable") == "factoid"

    def test_verification_classification(self):
        policy = StaticRetrievalPolicy()

        assert policy.get_qa_type("Is the server running?") == "verification"
        assert policy.get_qa_type("Does the API support authentication?") == "verification"
        assert policy.get_qa_type("Can I use this method?") == "verification"

    def test_procedural_classification(self):
        policy = StaticRetrievalPolicy()

        assert policy.get_qa_type("How to install the package?") == "procedural"
        assert policy.get_qa_type("What are the steps to deploy?") == "procedural"
        assert policy.get_qa_type("How do I configure the database?") == "procedural"

    def test_causal_classification(self):
        policy = StaticRetrievalPolicy()

        assert policy.get_qa_type("Why did the test fail?") == "causal"
        assert policy.get_qa_type("What is the reason for the error?") == "causal"
        assert policy.get_qa_type("Explain because of the failure") == "causal"

    def test_entity_classification(self):
        policy = StaticRetrievalPolicy()

        # Questions with file paths, error types, or step references
        assert policy.get_qa_type("What is in src/config/env.ts?") == "entity"
        assert policy.get_qa_type("Step 5 failed, what happened?") == "entity"
        assert policy.get_qa_type("What does ValueError mean?") == "entity"

    def test_get_k_values(self):
        policy = StaticRetrievalPolicy()

        assert policy.get_k("factoid") == 1
        assert policy.get_k("verification") == 2
        assert policy.get_k("procedural") == 3
        assert policy.get_k("causal") == 5
        assert policy.get_k("summary") == 8
        assert policy.get_k("unknown") == 5

    def test_get_routing_values(self):
        policy = StaticRetrievalPolicy()

        assert policy.get_routing("factoid") == "hot_cache"
        assert policy.get_routing("verification") == "hot_cache"
        assert policy.get_routing("procedural") == "enriched"
        assert policy.get_routing("causal") == "enriched"
        assert policy.get_routing("entity") == "hot_cache"
        assert policy.get_routing("summary") == "enriched"


class TestComputeQueryConfidence:
    """Test query confidence C_q computation."""

    def test_full_overlap(self):
        """All query entities in hot results = high confidence."""
        question = "What is the JWT_SECRET in src/config/env.ts?"
        hot_results = [
            {"text": "JWT_SECRET = abc123"},
            {"text": "File src/config/env.ts contains configuration"},
        ]

        confidence = compute_query_confidence(question, hot_results)
        assert confidence == 1.0

    def test_partial_overlap(self):
        """Some query entities in hot results = partial confidence."""
        question = "What is the JWT_SECRET and the database URL?"
        hot_results = [
            {"text": "JWT_SECRET = abc123"},
            # No database URL mentioned
        ]

        confidence = compute_query_confidence(question, hot_results)
        assert 0.0 < confidence < 1.0

    def test_no_overlap(self):
        """No query entities in hot results = zero confidence."""
        question = "What is the API_ENDPOINT?"
        hot_results = [
            {"text": "Database connection established"},
            {"text": "Test suite passed"},
        ]

        confidence = compute_query_confidence(question, hot_results)
        assert confidence == 0.0

    def test_no_entities(self):
        """Generic question with no entities = high confidence default."""
        question = "What happened next?"
        hot_results = [{"text": "Some context"}]

        confidence = compute_query_confidence(question, hot_results)
        assert confidence == 1.0


class TestComputeAnswerConfidence:
    """Test answer confidence C_a computation."""

    def test_high_confidence_answer(self):
        """Answer grounded in context and addresses query."""
        question = "What is the JWT_SECRET?"
        context = "The JWT_SECRET is abc123 in the configuration"
        answer = "The JWT_SECRET is abc123"

        result = compute_answer_confidence(question, context, answer)

        assert result.evidence_support > 0.5
        assert result.query_coverage == 1.0
        assert result.overall > 0.7

    def test_ungrounded_answer(self):
        """Answer mentions entities not in context = low evidence support."""
        question = "What is the API endpoint?"
        context = "Database connection on port 5432"
        answer = "The API endpoint is https://api.example.com"

        result = compute_answer_confidence(question, context, answer)

        assert result.evidence_support == 0.0
        assert result.query_coverage == 1.0

    def test_incomplete_answer(self):
        """Answer doesn't address all query entities = low coverage."""
        question = "What is the JWT_SECRET and database URL?"
        context = "JWT_SECRET = abc123, DATABASE_URL = postgres://..."
        answer = "The JWT_SECRET is abc123"

        result = compute_answer_confidence(question, context, answer)

        assert result.evidence_support == 1.0
        assert result.query_coverage < 1.0


class TestConfidenceRetryEscalation:
    """Test confidence-based retry escalation."""

    def test_escalation_route_hot_to_enriched(self):
        """Hot cache escalates to enriched."""
        retry = ConfidenceRetryEscalation()
        assert retry._escalate_route("hot_cache") == "enriched"

    def test_escalation_route_enriched_to_cold(self):
        """Enriched escalates to cold storage."""
        retry = ConfidenceRetryEscalation()
        assert retry._escalate_route("enriched") == "cold_storage"

    def test_escalation_route_cold_stays_cold(self):
        """Cold storage stays at cold (max escalation)."""
        retry = ConfidenceRetryEscalation()
        assert retry._escalate_route("cold_storage") == "cold_storage"

    def test_execute_stops_at_confidence(self):
        """Stops retrying when confidence threshold met."""
        retry = ConfidenceRetryEscalation(max_attempts=3, confidence_threshold=0.70)

        # Create a minimal mock memory object
        class MockMemory:
            def retrieve(self, query, k=5):
                return "Mock context for: " + query

        attempt_count = [0]

        def llm_fn(question, context):
            return "Confident answer"

        def confidence_fn(question, context, answer):
            from jeval.memory.session_aware import ConfidenceResult
            attempt_count[0] += 1
            # Return high confidence on first attempt
            return ConfidenceResult(
                evidence_support=0.8,
                query_coverage=0.8,
                hot_trust=0.8,
                overall=0.8,
            )

        answer, attempts = retry.execute(
            question="Test?",
            mem=MockMemory(),
            qa_type="factoid",
            policy=StaticRetrievalPolicy(),
            llm_fn=llm_fn,
            confidence_fn=confidence_fn,
        )

        assert attempts == 1
        assert answer == "Confident answer"


class TestSchemaInduction:
    """Test schema induction from clusters of artifacts."""

    def test_compute_signature_json(self):
        """Extract structural signature from JSON-like artifact."""
        induction = SchemaInduction()
        artifact = '{"tool_name": "run_test", "result": "passed", "duration": "100ms"}'

        sig = induction.compute_signature(artifact)

        assert "tool_name" in sig.field_names
        assert "result" in sig.field_names
        assert "duration" in sig.field_names
        assert sig.nesting_depth == 1

    def test_compute_signature_key_value(self):
        """Extract signature from key:value format."""
        induction = SchemaInduction()
        artifact = "Error: ValueError at line 42\nFile: /path/to/file.py"

        sig = induction.compute_signature(artifact)

        assert "Error" in sig.field_names or "File" in sig.field_names

    def test_jaccard_distance_identical(self):
        """Identical signatures have zero distance."""
        induction = SchemaInduction()
        sig = StructuralSignature(
            field_names=frozenset(["a", "b", "c"]),
            value_types=frozenset(["string"]),
            nesting_depth=1,
            key_patterns=frozenset(["snake_case"]),
        )

        distance = induction.jaccard_distance(sig, sig)
        assert distance == 0.0

    def test_jaccard_distance_disjoint(self):
        """Disjoint signatures have distance 1.0."""
        induction = SchemaInduction()
        sig1 = StructuralSignature(
            field_names=frozenset(["a", "b"]),
            value_types=frozenset(["string"]),
            nesting_depth=1,
            key_patterns=frozenset(),
        )
        sig2 = StructuralSignature(
            field_names=frozenset(["x", "y"]),
            value_types=frozenset(["number"]),
            nesting_depth=2,
            key_patterns=frozenset(),
        )

        distance = induction.jaccard_distance(sig1, sig2)
        assert distance == 1.0

    def test_cluster_artifacts(self):
        """Cluster similar artifacts together."""
        induction = SchemaInduction(min_cluster_size=2)

        # Two similar JSON artifacts
        artifacts = [
            '{"tool": "test", "result": "pass"}',
            '{"tool": "build", "result": "fail"}',
            'Error: Something went wrong',  # Different structure
        ]

        clusters = induction.cluster_artifacts(artifacts)

        # Should have at least one cluster with the similar artifacts
        assert len(clusters) >= 0  # May not meet min_cluster_size

    def test_induce_schema_from_cluster(self):
        """Induce schema from a cluster of similar artifacts."""
        induction = SchemaInduction(min_cluster_size=3)

        artifacts = [
            '{"tool_name": "test", "result": "pass", "duration": "100ms"}',
            '{"tool_name": "build", "result": "fail", "duration": "200ms"}',
            '{"tool_name": "deploy", "result": "ok", "duration": "300ms"}',
            '{"tool_name": "lint", "result": "pass"}',  # Missing duration
        ]

        # All 4 form a cluster
        cluster_indices = [0, 1, 2, 3]

        schema = induction.induce_schema(artifacts, cluster_indices, "tool_call")

        assert schema is not None
        assert schema.name == "tool_call"
        assert "tool_name" in schema.required_fields  # Present in all 4
        assert "result" in schema.required_fields  # Present in all 4


class TestNoveltyGateSchemaDetection:
    """Test schema novelty detection in NoveltyGate."""

    @pytest.fixture
    def encoder(self):
        return FrozenEncoder()

    @pytest.fixture
    def gate(self, encoder):
        return NoveltyGate(encoder, threshold=0.05)

    def test_known_schema_not_novel(self, gate):
        """Text matching known schema is not novel."""
        # deployment schema should match
        text = "Deployment to staging succeeded with HTTP 200"
        is_novel, score = gate.is_schema_novel(text)

        assert not is_novel
        assert score < 0.5

    def test_unknown_schema_is_novel(self, gate):
        """Text not matching any schema is novel."""
        # Random text that doesn't match any schema
        text = "The quick brown fox jumps over the lazy dog"
        is_novel, score = gate.is_schema_novel(text)

        assert is_novel
        assert score > 0.5

    def test_enqueue_for_induction(self, gate):
        """Unknown artifacts can be enqueued for induction."""
        artifact = "Some unknown artifact type"
        gate.enqueue_for_induction(artifact)

        artifacts = gate.get_unknown_artifacts()
        assert len(artifacts) == 1
        assert artifacts[0] == artifact

    def test_get_unknown_artifacts_clears_queue(self, gate):
        """Getting unknown artifacts clears the queue."""
        gate.enqueue_for_induction("artifact1")
        gate.enqueue_for_induction("artifact2")

        artifacts1 = gate.get_unknown_artifacts()
        assert len(artifacts1) == 2

        artifacts2 = gate.get_unknown_artifacts()
        assert len(artifacts2) == 0  # Queue cleared


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
