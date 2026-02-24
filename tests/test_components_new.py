"""Tests for new component dataclasses in definitions.py."""

import pytest

from ecs_agent.components.definitions import (
    EmbeddingComponent,
    PlanSearchComponent,
    RAGTriggerComponent,
    SandboxConfigComponent,
    ToolApprovalComponent,
    VectorStoreComponent,
)
from ecs_agent.types import ApprovalPolicy


class TestToolApprovalComponent:
    """Test ToolApprovalComponent instantiation and fields."""

    def test_instantiate_with_policy_only(self) -> None:
        """Test instantiation with required policy argument."""
        comp = ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL)
        assert comp.policy == ApprovalPolicy.REQUIRE_APPROVAL
        assert comp.timeout == 30.0
        assert comp.approved_calls == []
        assert comp.denied_calls == []

    def test_instantiate_with_all_args(self) -> None:
        """Test instantiation with all arguments."""
        comp = ToolApprovalComponent(
            policy=ApprovalPolicy.ALWAYS_APPROVE,
            timeout=60.0,
            approved_calls=["tool1", "tool2"],
            denied_calls=["tool3"],
        )
        assert comp.policy == ApprovalPolicy.ALWAYS_APPROVE
        assert comp.timeout == 60.0
        assert comp.approved_calls == ["tool1", "tool2"]
        assert comp.denied_calls == ["tool3"]

    def test_has_slots(self) -> None:
        """Test that ToolApprovalComponent has __slots__."""
        assert hasattr(ToolApprovalComponent, "__slots__")

    def test_timeout_field_type(self) -> None:
        """Test timeout field is float."""
        comp = ToolApprovalComponent(policy=ApprovalPolicy.ALWAYS_DENY, timeout=15.5)
        assert isinstance(comp.timeout, float)

    def test_mutable_defaults_independent(self) -> None:
        """Test mutable defaults are independent between instances."""
        comp1 = ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL)
        comp2 = ToolApprovalComponent(policy=ApprovalPolicy.REQUIRE_APPROVAL)
        comp1.approved_calls.append("tool1")
        assert comp2.approved_calls == []


class TestSandboxConfigComponent:
    """Test SandboxConfigComponent instantiation and fields."""

    def test_instantiate_defaults(self) -> None:
        """Test default instantiation."""
        comp = SandboxConfigComponent()
        assert comp.timeout == 30.0
        assert comp.max_output_size == 10_000

    def test_instantiate_with_args(self) -> None:
        """Test instantiation with custom arguments."""
        comp = SandboxConfigComponent(timeout=45.0, max_output_size=50_000)
        assert comp.timeout == 45.0
        assert comp.max_output_size == 50_000

    def test_has_slots(self) -> None:
        """Test that SandboxConfigComponent has __slots__."""
        assert hasattr(SandboxConfigComponent, "__slots__")

    def test_timeout_field_type(self) -> None:
        """Test timeout field is float."""
        comp = SandboxConfigComponent(timeout=20.5)
        assert isinstance(comp.timeout, float)

    def test_max_output_size_field_type(self) -> None:
        """Test max_output_size field is int."""
        comp = SandboxConfigComponent(max_output_size=20_000)
        assert isinstance(comp.max_output_size, int)


class TestPlanSearchComponent:
    """Test PlanSearchComponent instantiation and fields."""

    def test_instantiate_defaults(self) -> None:
        """Test default instantiation."""
        comp = PlanSearchComponent()
        assert comp.max_depth == 5
        assert comp.max_branching == 3
        assert abs(comp.exploration_weight - 1.414) < 0.001
        assert comp.best_plan == []
        assert comp.search_active is False

    def test_instantiate_with_args(self) -> None:
        """Test instantiation with custom arguments."""
        comp = PlanSearchComponent(
            max_depth=10,
            max_branching=5,
            exploration_weight=2.0,
            best_plan=["step1", "step2"],
            search_active=True,
        )
        assert comp.max_depth == 10
        assert comp.max_branching == 5
        assert comp.exploration_weight == 2.0
        assert comp.best_plan == ["step1", "step2"]
        assert comp.search_active is True

    def test_has_slots(self) -> None:
        """Test that PlanSearchComponent has __slots__."""
        assert hasattr(PlanSearchComponent, "__slots__")

    def test_best_plan_mutable_defaults_independent(self) -> None:
        """Test best_plan mutable defaults are independent between instances."""
        comp1 = PlanSearchComponent()
        comp2 = PlanSearchComponent()
        comp1.best_plan.append("step1")
        assert comp2.best_plan == []

    def test_field_types(self) -> None:
        """Test field types are correct."""
        comp = PlanSearchComponent()
        assert isinstance(comp.max_depth, int)
        assert isinstance(comp.max_branching, int)
        assert isinstance(comp.exploration_weight, float)
        assert isinstance(comp.best_plan, list)
        assert isinstance(comp.search_active, bool)


class TestRAGTriggerComponent:
    """Test RAGTriggerComponent instantiation and fields."""

    def test_instantiate_defaults(self) -> None:
        """Test default instantiation."""
        comp = RAGTriggerComponent()
        assert comp.query == ""
        assert comp.top_k == 5
        assert comp.retrieved_docs == []

    def test_instantiate_with_args(self) -> None:
        """Test instantiation with custom arguments."""
        comp = RAGTriggerComponent(
            query="test query",
            top_k=10,
            retrieved_docs=["doc1", "doc2"],
        )
        assert comp.query == "test query"
        assert comp.top_k == 10
        assert comp.retrieved_docs == ["doc1", "doc2"]

    def test_has_slots(self) -> None:
        """Test that RAGTriggerComponent has __slots__."""
        assert hasattr(RAGTriggerComponent, "__slots__")

    def test_retrieved_docs_mutable_defaults_independent(self) -> None:
        """Test retrieved_docs mutable defaults are independent between instances."""
        comp1 = RAGTriggerComponent()
        comp2 = RAGTriggerComponent()
        comp1.retrieved_docs.append("doc1")
        assert comp2.retrieved_docs == []

    def test_field_types(self) -> None:
        """Test field types are correct."""
        comp = RAGTriggerComponent()
        assert isinstance(comp.query, str)
        assert isinstance(comp.top_k, int)
        assert isinstance(comp.retrieved_docs, list)


class TestEmbeddingComponent:
    """Test EmbeddingComponent instantiation and fields."""

    def test_instantiate_with_provider(self) -> None:
        """Test instantiation with provider."""
        provider = object()
        comp = EmbeddingComponent(provider=provider)
        assert comp.provider is provider
        assert comp.dimension == 0

    def test_instantiate_with_all_args(self) -> None:
        """Test instantiation with all arguments."""
        provider = object()
        comp = EmbeddingComponent(provider=provider, dimension=384)
        assert comp.provider is provider
        assert comp.dimension == 384

    def test_has_slots(self) -> None:
        """Test that EmbeddingComponent has __slots__."""
        assert hasattr(EmbeddingComponent, "__slots__")

    def test_dimension_field_type(self) -> None:
        """Test dimension field is int."""
        comp = EmbeddingComponent(provider=None, dimension=768)
        assert isinstance(comp.dimension, int)


class TestVectorStoreComponent:
    """Test VectorStoreComponent instantiation and fields."""

    def test_instantiate_with_store(self) -> None:
        """Test instantiation with store."""
        store = object()
        comp = VectorStoreComponent(store=store)
        assert comp.store is store

    def test_instantiate_with_none(self) -> None:
        """Test instantiation with None store."""
        comp = VectorStoreComponent(store=None)
        assert comp.store is None

    def test_has_slots(self) -> None:
        """Test that VectorStoreComponent has __slots__."""
        assert hasattr(VectorStoreComponent, "__slots__")


class TestAllComponentsSlots:
    """Meta test verifying all new components have __slots__."""

    @pytest.mark.parametrize(
        "component_class",
        [
            ToolApprovalComponent,
            SandboxConfigComponent,
            PlanSearchComponent,
            RAGTriggerComponent,
            EmbeddingComponent,
            VectorStoreComponent,
        ],
    )
    def test_all_components_have_slots(self, component_class: type) -> None:
        """Test that all new components have __slots__."""
        assert hasattr(component_class, "__slots__"), (
            f"{component_class.__name__} missing __slots__"
        )
