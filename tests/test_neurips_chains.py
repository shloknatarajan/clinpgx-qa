"""Tests for neurips_chains module."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_generation"))

from neurips_chains import (
    QuestionChain,
    Turn,
    build_chain_a,
    build_chain_b,
    build_chain_c,
    build_chain_e,
    build_swap_pools,
    is_p_significant,
    make_negative,
    parse_ci,
    parse_p_value,
    safe_float,
    safe_int,
    validate_chains,
)

# ═════════════════════════════════════════════════════════════════════════════
# P-Value Parsing
# ═════════════════════════════════════════════════════════════════════════════


class TestParsePValue:
    def test_bare_numeric(self):
        comp, val = parse_p_value("0.03")
        assert comp == "="
        assert val == pytest.approx(0.03)

    def test_equals_prefix(self):
        comp, val = parse_p_value("= 0.231")
        assert comp == "="
        assert val == pytest.approx(0.231)

    def test_less_than_prefix(self):
        comp, val = parse_p_value("< 0.001")
        assert comp == "<"
        assert val == pytest.approx(0.001)

    def test_greater_than_prefix(self):
        comp, val = parse_p_value("> 0.05")
        assert comp == ">"
        assert val == pytest.approx(0.05)

    def test_empty_string(self):
        comp, val = parse_p_value("")
        assert comp == ""
        assert val is None

    def test_whitespace(self):
        comp, val = parse_p_value("  = 0.05  ")
        assert comp == "="
        assert val == pytest.approx(0.05)

    def test_non_numeric(self):
        comp, val = parse_p_value("= abc")
        assert comp == "="
        assert val is None


class TestIsPSignificant:
    def test_significant_equals(self):
        assert is_p_significant("= 0.03") is True

    def test_not_significant_equals(self):
        assert is_p_significant("= 0.06") is False

    def test_significant_less_than(self):
        assert is_p_significant("< 0.001") is True

    def test_less_than_at_boundary(self):
        assert is_p_significant("< 0.05") is True

    def test_greater_than_above_alpha(self):
        assert is_p_significant("> 0.05") is False

    def test_empty(self):
        assert is_p_significant("") is None

    def test_custom_alpha(self):
        assert is_p_significant("= 0.03", alpha=0.01) is False
        assert is_p_significant("= 0.005", alpha=0.01) is True

    def test_bare_numeric_significant(self):
        assert is_p_significant("0.04") is True

    def test_bare_numeric_not_significant(self):
        assert is_p_significant("0.06") is False


class TestParseCI:
    def test_valid(self):
        s, e = parse_ci("0.5", "1.5")
        assert s == pytest.approx(0.5)
        assert e == pytest.approx(1.5)

    def test_empty(self):
        s, e = parse_ci("", "")
        assert s is None
        assert e is None

    def test_partial(self):
        s, e = parse_ci("0.5", "")
        assert s == pytest.approx(0.5)
        assert e is None


class TestSafeConversions:
    def test_safe_int_normal(self):
        assert safe_int("42") == 42

    def test_safe_int_float_string(self):
        assert safe_int("30.0") == 30

    def test_safe_int_empty(self):
        assert safe_int("") is None

    def test_safe_int_non_numeric(self):
        assert safe_int("abc") is None

    def test_safe_float_normal(self):
        assert safe_float("1.5") == pytest.approx(1.5)

    def test_safe_float_empty(self):
        assert safe_float("") is None


# ═════════════════════════════════════════════════════════════════════════════
# Chain Coherence Rules
# ═════════════════════════════════════════════════════════════════════════════

SAMPLE_ANN = {
    "Variant Annotation ID": "123",
    "Variant/Haplotypes": "rs12345",
    "Gene": "CYP2D6",
    "Drug(s)": "tamoxifen",
    "PMID": "99999",
    "Phenotype Category": "Efficacy",
    "Significance": "yes",
    "Notes": "",
    "Sentence": "rs12345 is associated with increased response to tamoxifen.",
    "Is/Is Not associated": "Associated with",
    "Direction of effect": "increased",
    "PD/PK terms": "response to",
}

SAMPLE_SP = {
    "Study Parameters ID": "456",
    "Variant Annotation ID": "123",
    "Study Type": "case/control",
    "Study Cases": "100",
    "Study Controls": "200",
    "P Value": "= 0.01",
    "Ratio Stat Type": "OR",
    "Ratio Stat": "2.5",
    "Confidence Interval Start": "1.2",
    "Confidence Interval Stop": "5.1",
    "Biogeographical Groups": "European",
}


class TestFamilyBNoStats:
    def test_family_b_has_no_statistical_extraction(self):
        cand = (SAMPLE_ANN, "var_drug_ann", "Fake claim not reported.", "not_reported_variant_swap")
        rng = random.Random(42)
        chain = build_chain_b(cand, "chain_test", rng, {}, False)
        for t in chain.turns:
            assert t.reasoning_type != "statistical_extraction", (
                "Family B must not contain statistical_extraction turns"
            )

    def test_family_b_has_negative(self):
        cand = (SAMPLE_ANN, "var_drug_ann", "Fake claim.", "not_reported_drug_swap")
        rng = random.Random(42)
        chain = build_chain_b(cand, "chain_test", rng, {}, False)
        assert chain.has_negative is True
        assert any(t.negative_type is not None for t in chain.turns)


class TestFamilyAStructure:
    def test_family_a_has_4_turns(self):
        cand = (SAMPLE_ANN, "var_drug_ann", SAMPLE_SP, "P Value")
        rng = random.Random(42)
        chain = build_chain_a(cand, "chain_test", rng, {}, False)
        assert chain.num_turns == 4
        assert len(chain.turns) == 4
        assert [t.turn for t in chain.turns] == [1, 2, 3, 4]

    def test_family_a_turn_types(self):
        cand = (SAMPLE_ANN, "var_drug_ann", SAMPLE_SP, "P Value")
        rng = random.Random(42)
        chain = build_chain_a(cand, "chain_test", rng, {}, False)
        assert chain.turns[0].reasoning_type == "claim_verification"
        assert chain.turns[1].reasoning_type == "evidence_provenance_localization"
        assert chain.turns[2].reasoning_type == "statistical_extraction"
        assert chain.turns[3].reasoning_type in (
            "objective_evaluation", "counterfactual_evaluation",
        )


# ═════════════════════════════════════════════════════════════════════════════
# Negative Swap Does Not Reproduce Original
# ═════════════════════════════════════════════════════════════════════════════


class TestNegativeSwap:
    def test_swap_does_not_reproduce_original(self):
        all_anns = [
            (SAMPLE_ANN, "var_drug_ann"),
            (
                {
                    **SAMPLE_ANN,
                    "Variant/Haplotypes": "rs99999",
                    "PMID": "11111",
                    "Sentence": "rs99999 is associated with decreased response to tamoxifen.",
                },
                "var_drug_ann",
            ),
        ]
        pools = build_swap_pools(all_anns, {})
        rng = random.Random(42)
        result = make_negative(SAMPLE_ANN, "var_drug_ann", pools, rng)
        assert result is not None
        neg_sentence, neg_type = result
        assert neg_sentence != SAMPLE_ANN["Sentence"]
        assert "not_reported" in neg_type

    def test_no_swap_when_alone(self):
        all_anns = [(SAMPLE_ANN, "var_drug_ann")]
        pools = build_swap_pools(all_anns, {})
        rng = random.Random(42)
        # Only phenotype swap possible (pool has only one gene/pheno combo with one variant)
        result = make_negative(SAMPLE_ANN, "var_drug_ann", pools, rng)
        # Should still produce a phenotype swap if other phenotypes exist
        # Since pool has only "Efficacy", phenotype swap won't work either
        # but the pool from a single annotation still has the phenotype itself
        # so alt_phenos will be empty → no options → None is valid
        # Actually phenotypes set = {"Efficacy"}, alt_phenos = [] → no swap possible
        # Unless there are other phenotype categories. With single ann, only "Efficacy".
        # So result should be None
        assert result is None


# ═════════════════════════════════════════════════════════════════════════════
# Validation
# ═════════════════════════════════════════════════════════════════════════════


class TestValidation:
    def test_valid_chain_passes(self):
        cand = (SAMPLE_ANN, "var_drug_ann", SAMPLE_SP, "P Value")
        rng = random.Random(42)
        chain = build_chain_a(cand, "chain_000001", rng, {}, False)
        validate_chains([chain])

    def test_invalid_turn_numbers_fail(self):
        cand = (SAMPLE_ANN, "var_drug_ann", SAMPLE_SP, "P Value")
        rng = random.Random(42)
        chain = build_chain_a(cand, "chain_000001", rng, {}, False)
        chain.turns[2] = chain.turns[2].model_copy(update={"turn": 99})
        with pytest.raises(ValueError, match="Validation failed"):
            validate_chains([chain])

    def test_family_b_with_stat_fails(self):
        cand = (SAMPLE_ANN, "var_drug_ann", "Fake claim.", "not_reported_drug_swap")
        rng = random.Random(42)
        chain = build_chain_b(cand, "chain_test", rng, {}, False)
        # Manually inject a bad turn
        bad_turn = Turn(
            turn=3,
            reasoning_type="statistical_extraction",
            question="bad",
            answer="bad",
            answer_source_fields=["P Value"],
            evidence_required=True,
            evidence_granularity="db_record",
        )
        chain.turns.append(bad_turn)
        chain.num_turns = 3
        with pytest.raises(ValueError, match="statistical_extraction"):
            validate_chains([chain])


class TestFamilyCStructure:
    def test_family_c_base_3_turns(self):
        summary = {
            "Summary Annotation ID": "S1",
            "Variant/Haplotypes": "rs111",
            "Gene": "CYP2C19",
            "Drug(s)": "clopidogrel",
            "Level of Evidence": "1A",
            "PMID Count": "5",
        }
        cand = (summary, [], False)
        rng = random.Random(42)
        chain = build_chain_c(cand, "chain_c", rng)
        assert chain.num_turns == 3
        assert chain.turns[0].reasoning_type == "evidence_aggregation"
        assert chain.turns[2].answer is True  # 5 > 1


class TestFamilyEStructure:
    def test_family_e_2_turns(self):
        summary = {
            "Summary Annotation ID": "S2",
            "Variant/Haplotypes": "rs222",
            "Gene": "VKORC1",
            "Latest History Date (YYYY-MM-DD)": "2023-01-15",
        }
        history = [
            {"Date (YYYY-MM-DD)": "2021-03-24", "Type": "Update", "Comment": "Score added"},
            {"Date (YYYY-MM-DD)": "2023-01-15", "Type": "Correction", "Comment": "Fixed"},
        ]
        cand = (summary, history, 2020)
        rng = random.Random(42)
        chain = build_chain_e(cand, "chain_e", rng)
        assert chain.num_turns == 2
        assert chain.turns[0].answer is True  # 2023 > 2020
        assert chain.turns[1].answer == 2  # 2 updates/corrections
