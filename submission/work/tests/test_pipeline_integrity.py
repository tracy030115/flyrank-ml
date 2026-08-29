"""Data validation tests against the interim CSV hand-offs the five notebooks produce.

These are integration/regression tests, not unit tests: they read the actual files under
work/interim/ and assert the properties the whole project's methodology depends on, most
importantly zero client overlap between train and test, which is the entire point of the
client-grouped split. If a future change to any notebook breaks one of these properties,
these tests fail without needing to re-read every notebook's printed output by eye.

Requires work/interim/*.csv to already exist, i.e. the five notebooks have been run at
least once. Tests skip cleanly rather than fail if the interim files are not present.
"""

import os

import pandas as pd
import pytest

INTERIM_DIR = os.path.join(os.path.dirname(__file__), "..", "interim")

# the five behavioral features the final KMeans model clusters on, kept in sync by hand
# with the same list inlined in each of the five notebooks
FEATURE_COLS = ["ctr", "avg_position", "days_since_last_update", "total_impressions_log", "engagement_rate_filled"]

# kept in sync by hand with the same mapping inlined in the baseline and clustering
# notebooks, so the same archetype always maps to the same action everywhere in the project
ACTION_MAP = {
    "champions": "protect",
    "hidden_gems": "improve",
    "rising_stars": "monitor",
    "stale_visible_pages": "refresh",
    "engagement_problem_pages": "rewrite",
    "weak_no_demand_pages": "prune",
    "cannibalization_risk": "merge",
    "monitor": "monitor",
}

# columns that must never be used as clustering inputs, since each one either encodes the
# rule based label directly or is a downstream identifier the model should not see
FORBIDDEN_FEATURE_COLUMNS = ["archetype", "cannibalization_risk", "keyword_hash_id"]

KNOWN_ARCHETYPE_KMEANS_VALUES = {
    "champions", "hidden_gems", "rising_stars", "stale_visible_pages",
    "engagement_problem_pages", "weak_no_demand_pages", "needs_review",
}


def _interim_path(filename):
    return os.path.join(INTERIM_DIR, filename)


def _require_csv(filename):
    path = _interim_path(filename)
    if not os.path.exists(path):
        pytest.skip(f"{filename} not found under work/interim/; run the notebooks first")
    return pd.read_csv(path, low_memory=False)


@pytest.fixture(scope="module")
def page_level():
    return _require_csv("page_level.csv")


@pytest.fixture(scope="module")
def train():
    return _require_csv("train.csv")


@pytest.fixture(scope="module")
def test_df():
    return _require_csv("test.csv")


@pytest.fixture(scope="module")
def best_archetype_table():
    return _require_csv("best_archetype_table.csv")


# ---------------------------------------------------------------------------
# FEATURE_COLS itself, no CSV needed
# ---------------------------------------------------------------------------

class TestFeatureColsNeverLeak:
    def test_no_forbidden_column_in_feature_cols(self):
        for forbidden in FORBIDDEN_FEATURE_COLUMNS:
            assert forbidden not in FEATURE_COLS

    def test_feature_cols_is_the_expected_five(self):
        assert FEATURE_COLS == [
            "ctr", "avg_position", "days_since_last_update",
            "total_impressions_log", "engagement_rate_filled",
        ]


# ---------------------------------------------------------------------------
# page_level.csv, the rule based baseline's output
# ---------------------------------------------------------------------------

class TestPageLevel:
    def test_row_count(self, page_level):
        assert len(page_level) == 394928

    def test_negative_staleness_rows_are_all_insufficient_data(self, page_level):
        # the fix does not scrub days_since_last_update itself, it routes any row
        # carrying a negative value to INSUFFICIENT_DATA so it never reaches a real
        # archetype branch. page_level.csv legitimately retains the raw negative values;
        # what must hold is that none of those rows got a real archetype
        negative_rows = page_level[page_level["days_since_last_update"] < 0]
        if len(negative_rows) > 0:
            assert (negative_rows["archetype"] == "INSUFFICIENT_DATA").all()

    def test_no_negative_staleness_outside_insufficient_data(self, page_level):
        usable = page_level[page_level["archetype"] != "INSUFFICIENT_DATA"]
        assert (usable["days_since_last_update"] < 0).sum() == 0

    def test_archetype_value_counts_sum_to_total_rows(self, page_level):
        assert page_level["archetype"].value_counts().sum() == len(page_level)

    def test_every_archetype_has_a_known_action(self, page_level):
        archetypes_present = set(page_level["archetype"].unique()) - {"INSUFFICIENT_DATA"}
        for archetype in archetypes_present:
            assert archetype in ACTION_MAP, f"{archetype} has no entry in ACTION_MAP"

    def test_action_score_is_never_negative(self, page_level):
        assert (page_level["action_score"] >= 0).all()

    def test_cannibalization_rows_are_rare(self, page_level):
        # a sanity bound, not a business rule: this population has had exactly 2
        # cannibalization_risk rows throughout the project; a sudden jump would be worth
        # investigating rather than silently accepting
        n_cannibalization = (page_level["archetype"] == "cannibalization_risk").sum()
        assert 0 <= n_cannibalization < len(page_level) * 0.01


# ---------------------------------------------------------------------------
# train.csv / test.csv, the client-grouped split
# ---------------------------------------------------------------------------

class TestClientGroupedSplit:
    def test_zero_client_overlap(self, train, test_df):
        # this is the single most important property in the entire project: the client
        # grouped split exists specifically to make this assertion true
        overlap = set(train["client_hash_id"]) & set(test_df["client_hash_id"])
        assert len(overlap) == 0

    def test_row_counts(self, train, test_df):
        assert len(train) == 107523
        assert len(test_df) == 14856

    def test_kmeans_cluster_ids_in_expected_range(self, train, test_df):
        assert train["kmeans_cluster"].min() >= 0
        assert train["kmeans_cluster"].max() == 19  # k=20, ids 0 through 19
        assert set(test_df["kmeans_cluster"].unique()) <= set(range(20))

    def test_archetype_kmeans_values_are_all_known(self, train, test_df):
        assert set(train["archetype_kmeans"].unique()) <= KNOWN_ARCHETYPE_KMEANS_VALUES
        assert set(test_df["archetype_kmeans"].unique()) <= KNOWN_ARCHETYPE_KMEANS_VALUES

    def test_recommended_action_values_are_all_known(self, train):
        known_actions = set(ACTION_MAP.values())
        assert set(train["recommended_action"].unique()) <= known_actions

    def test_needs_review_maps_to_monitor_unless_cannibalizing(self, train):
        # a needs_review page's action is "monitor" by default; the one exception is a
        # page also flagged cannibalization_risk by the independent rule engine, which
        # overrides the action to "merge" regardless of how the cluster named it
        needs_review_rows = train[train["archetype_kmeans"] == "needs_review"]
        non_cannibalizing = needs_review_rows[~needs_review_rows["cannibalization_risk"]]
        assert (non_cannibalizing["recommended_action"] == "monitor").all()

        cannibalizing = needs_review_rows[needs_review_rows["cannibalization_risk"]]
        if len(cannibalizing) > 0:
            assert (cannibalizing["recommended_action"] == "merge").all()


# ---------------------------------------------------------------------------
# best_archetype_table.csv, the per-cluster naming confidence table
# ---------------------------------------------------------------------------

class TestBestArchetypeTable:
    def test_one_row_per_cluster(self, best_archetype_table):
        assert len(best_archetype_table) == 20

    def test_naming_confidence_is_a_valid_percentage(self, best_archetype_table):
        assert (best_archetype_table["naming_confidence_pct"] >= 0).all()
        assert (best_archetype_table["naming_confidence_pct"] <= 100).all()

    def test_confident_flag_matches_the_threshold(self, best_archetype_table):
        # the confident column must always agree with a fresh 35% check against the
        # naming_confidence_pct column in the same row, not drift from it
        recomputed = best_archetype_table["naming_confidence_pct"] >= 35.0
        assert (best_archetype_table["confident"] == recomputed).all()

    def test_needs_review_only_when_not_confident(self, best_archetype_table):
        needs_review_rows = best_archetype_table[best_archetype_table["archetype"] == "needs_review"]
        assert (needs_review_rows["confident"] == False).all()  # noqa: E712

    def test_non_needs_review_rows_are_confident(self, best_archetype_table):
        named_rows = best_archetype_table[best_archetype_table["archetype"] != "needs_review"]
        assert (named_rows["confident"] == True).all()  # noqa: E712
