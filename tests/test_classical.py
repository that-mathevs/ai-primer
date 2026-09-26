"""Specification for primer.ml.classical: decision trees, random forests, gradient boosting, and when they still win."""

import numpy as np
import pytest

from primer.ml.classical import (
    DecisionTree,
    GradientBoosting,
    RandomForest,
    averaged_variance,
    best_split,
    boosting_curves,
    bootstrap_sample,
    candidate_splits,
    depth_sweep,
    engineer_features,
    entropy,
    forest_curve,
    gini,
    make_tabular,
    moons_split,
    out_of_bag_fraction,
    permutation_importance,
    spam_emails,
    split_impurity,
    tabular_showdown,
)
# The eight emails of the worked example: [links, known_sender] and 1 = spam.
X_SPAM, Y_SPAM, _ = spam_emails()
# The four houses of the boosting example: size and price.
SIZES = np.array([[1.0], [2.0], [3.0], [4.0]])
PRICES = np.array([1.0, 2.0, 6.0, 7.0])


def noisy_moons():
    # 300 training and 200 validation points from two noisy, interleaved half-moons.
    return moons_split()


class TestMeasuringImpurity:
    def test_given_four_spam_and_four_ham_the_gini_impurity_is_one_half(self):
        # 1 - (0.5² + 0.5²): two blind guesses from a 50/50 bag disagree half the time.
        assert gini(np.array([1, 1, 1, 1, 0, 0, 0, 0])) == pytest.approx(0.5)

    def test_given_a_pure_node_the_gini_impurity_is_zero(self):
        assert gini(np.array([0, 0, 0])) == 0.0

    def test_given_one_ham_and_four_spam_the_gini_impurity_is_0_32(self):
        # 1 - (0.2² + 0.8²) = 1 - 0.68
        assert gini(np.array([0, 1, 1, 1, 1])) == pytest.approx(0.32)

    def test_given_an_even_split_of_two_classes_the_entropy_is_one_bit(self):
        assert entropy(np.array([0, 1, 0, 1])) == pytest.approx(1.0)

    def test_given_one_ham_and_four_spam_the_entropy_is_about_0_72_bits(self):
        # -(0.2·log2 0.2 + 0.8·log2 0.8), worked by hand
        assert entropy(np.array([0, 1, 1, 1, 1])) == pytest.approx(0.7219, abs=1e-4)


class TestChoosingTheBestQuestion:
    def test_given_the_eight_emails_asking_known_sender_leaves_weighted_impurity_0_2(self):
        # left: 3 ham (Gini 0); right: 1 ham + 4 spam (Gini 0.32); 3/8·0 + 5/8·0.32 = 0.2
        known = X_SPAM[:, 1] > 0.5
        assert split_impurity(Y_SPAM[known], Y_SPAM[~known]) == pytest.approx(0.2)

    def test_given_the_eight_emails_asking_more_than_2_links_leaves_weighted_impurity_0_375(self):
        many = X_SPAM[:, 0] > 2
        assert split_impurity(Y_SPAM[many], Y_SPAM[~many]) == pytest.approx(0.375)

    def test_given_the_eight_emails_the_best_first_question_is_known_sender(self):
        split = best_split(X_SPAM, Y_SPAM)
        assert (split.feature, split.threshold, split.impurity) == (1, 0.5, pytest.approx(0.2))

    def test_given_the_eight_emails_there_are_six_questions_worth_trying(self):
        # links takes six distinct values, so five thresholds between them, plus one for known_sender.
        candidates = candidate_splits(X_SPAM, Y_SPAM)
        assert [(c.feature, c.threshold) for c in candidates] == [(0, 0.5), (0, 2.0), (0, 3.5), (0, 4.5), (0, 5.5), (1, 0.5)]

    def test_given_values_0_and_3_on_either_side_the_threshold_sits_halfway_at_1_5(self):
        X = np.array([[0.0], [0.0], [3.0], [3.0]])
        assert best_split(X, np.array([0, 0, 1, 1])).threshold == 1.5

    def test_given_a_node_that_is_already_pure_there_is_no_question_worth_asking(self):
        assert best_split(X_SPAM[:3], Y_SPAM[:3]) is None


class TestGrowingATree:
    def test_given_the_eight_emails_a_depth_two_tree_sorts_every_one_correctly(self):
        tree = DecisionTree(max_depth=2).fit(X_SPAM, Y_SPAM)
        assert tree.predict(X_SPAM).tolist() == Y_SPAM.tolist()

    def test_given_a_depth_limit_of_one_the_tree_is_a_single_question_with_two_leaves(self):
        tree = DecisionTree(max_depth=1).fit(X_SPAM, Y_SPAM)
        assert (tree.depth, tree.n_leaves) == (1, 2)

    def test_given_a_depth_limit_the_tree_never_grows_deeper(self):
        X, y, _, _ = noisy_moons()
        assert DecisionTree(max_depth=4).fit(X, y).depth <= 4

    def test_given_the_spam_tree_its_rules_read_as_plain_questions(self):
        tree = DecisionTree(max_depth=2).fit(X_SPAM, Y_SPAM)
        assert tree.rules(["links", "known_sender"])[0] == "if known_sender <= 0.5:"

    def test_given_any_input_the_predicted_class_probabilities_sum_to_one(self):
        X, y, X_val, _ = noisy_moons()
        proba = DecisionTree(max_depth=3).fit(X, y).predict_proba(X_val)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_given_features_rescaled_or_squashed_the_tree_makes_the_same_predictions(self):
        # A tree only asks "is this bigger than that?", and any order-keeping transform keeps every answer.
        X, y, X_val, _ = noisy_moons()
        plain = DecisionTree(max_depth=5).fit(X, y).predict(X_val)
        squashed = DecisionTree(max_depth=5).fit(np.exp(3 * X), y).predict(np.exp(3 * X_val))
        assert plain.tolist() == squashed.tolist()

    def test_given_the_same_data_the_first_question_matches_scikit_learn(self):
        tree_module = pytest.importorskip("sklearn.tree")
        X, y, _, _ = noisy_moons()
        theirs = tree_module.DecisionTreeClassifier(max_depth=1).fit(X, y).tree_
        ours = best_split(X, y)
        assert (ours.feature, ours.threshold) == (theirs.feature[0], pytest.approx(theirs.threshold[0]))


class TestOverfitting:
    def test_given_no_depth_limit_the_tree_memorises_its_training_set(self):
        X, y, _, _ = noisy_moons()
        tree = DecisionTree().fit(X, y)
        assert np.mean(tree.predict(X) == y) == 1.0

    def test_given_noisy_data_a_moderate_depth_beats_unlimited_depth_on_validation(self):
        rows = {r["depth"]: r for r in depth_sweep(depths=(4, None))}
        assert rows[4]["val_accuracy"] > rows[None]["val_accuracy"]

    def test_given_more_depth_training_accuracy_never_falls(self):
        train = [r["train_accuracy"] for r in depth_sweep(depths=(1, 2, 4, 8, None))]
        assert train == sorted(train)


class TestBagging:
    def test_given_eight_examples_the_chance_one_is_left_out_of_a_bootstrap_sample_is_34_percent(self):
        # (7/8)^8, by hand
        assert out_of_bag_fraction(8) == pytest.approx(0.3436, abs=1e-4)

    def test_given_a_large_dataset_the_left_out_share_approaches_one_over_e(self):
        assert out_of_bag_fraction(100_000) == pytest.approx(0.3679, abs=1e-4)

    def test_given_many_bootstrap_samples_the_share_left_out_matches_the_formula(self):
        rng = np.random.default_rng(0)
        left_out = [1 - len(set(bootstrap_sample(8, rng).tolist())) / 8 for _ in range(4000)]
        assert np.mean(left_out) == pytest.approx(0.3436, abs=0.01)

    def test_given_correlation_0_3_and_ten_trees_the_averaged_variance_is_0_37(self):
        # 0.3·1 + (1 - 0.3)·1/10
        assert averaged_variance(sigma2=1.0, rho=0.3, n_trees=10) == pytest.approx(0.37)

    def test_given_uncorrelated_trees_averaging_divides_the_variance_by_their_number(self):
        assert averaged_variance(sigma2=2.0, rho=0.0, n_trees=4) == pytest.approx(0.5)


class TestRandomForest:
    def test_given_noisy_data_a_forest_beats_a_single_deep_tree_on_validation(self):
        X, y, X_val, y_val = noisy_moons()
        tree = np.mean(DecisionTree().fit(X, y).predict(X_val) == y_val)
        forest = np.mean(RandomForest(n_trees=50).fit(X, y).predict(X_val) == y_val)
        assert forest > tree + 0.03

    def test_given_more_trees_validation_accuracy_ends_higher_than_with_one(self):
        rows = forest_curve(n_trees=(1, 50))
        assert rows[-1]["val_accuracy"] > rows[0]["val_accuracy"]

    def test_given_the_same_seed_the_forest_is_reproducible(self):
        X, y, X_val, _ = noisy_moons()
        a = RandomForest(n_trees=5, seed=3).fit(X, y).predict_proba(X_val)
        b = RandomForest(n_trees=5, seed=3).fit(X, y).predict_proba(X_val)
        assert np.array_equal(a, b)


class TestGradientBoosting:
    def test_given_the_four_houses_the_starting_guess_is_the_mean_price_of_4(self):
        model = GradientBoosting(n_rounds=0).fit(SIZES, PRICES)
        assert model.predict(SIZES).tolist() == [4.0, 4.0, 4.0, 4.0]

    def test_given_learning_rate_one_half_one_round_moves_the_guesses_to_2_75_and_5_25(self):
        # Residuals -3, -2, 2, 3; the stump predicts -2.5 small, +2.5 big; half of that is added.
        model = GradientBoosting(n_rounds=1, learning_rate=0.5, max_depth=1).fit(SIZES, PRICES)
        assert model.predict(SIZES).tolist() == [2.75, 2.75, 5.25, 5.25]

    def test_given_that_round_the_mean_squared_error_falls_from_6_5_to_1_8125(self):
        model = GradientBoosting(n_rounds=1, learning_rate=0.5, max_depth=1).fit(SIZES, PRICES)
        assert model.train_loss == pytest.approx([6.5, 1.8125])

    def test_given_squared_loss_the_residual_is_the_negative_slope_of_the_loss(self):
        # Nudge the guess F and watch half the squared error change: the slope is F - y.
        y, F, eps = 7.0, 4.0, 1e-6
        slope = ((0.5 * (y - (F + eps)) ** 2) - (0.5 * (y - (F - eps)) ** 2)) / (2 * eps)
        assert GradientBoosting.negative_gradient(np.array([y]), np.array([F]), "squared")[0] == pytest.approx(-slope)

    def test_given_more_rounds_training_loss_never_rises(self):
        X, y, _, _ = noisy_moons()
        loss = GradientBoosting(n_rounds=30, learning_rate=0.3).fit(X, y.astype(float)).train_loss
        assert all(b <= a + 1e-12 for a, b in zip(loss, loss[1:]))

    def test_given_a_smaller_learning_rate_training_loss_falls_more_slowly(self):
        X, y, _, _ = noisy_moons()
        slow = GradientBoosting(n_rounds=10, learning_rate=0.1).fit(X, y.astype(float)).train_loss[-1]
        fast = GradientBoosting(n_rounds=10, learning_rate=0.5).fit(X, y.astype(float)).train_loss[-1]
        assert slow > fast

    def test_given_too_many_rounds_validation_loss_climbs_back_up_from_its_best(self):
        # Train loss heads for 0 on 80 noisy points; the extra rounds are fitting noise.
        row = boosting_curves(learning_rates=(1.0,))[0]
        assert row["val_loss"][-1] > min(row["val_loss"]) * 1.2

    def test_given_a_smaller_learning_rate_the_best_validation_loss_is_lower_and_arrives_later(self):
        fast, slow = boosting_curves(learning_rates=(1.0, 0.1))
        assert min(slow["val_loss"]) < min(fast["val_loss"])
        assert slow["best_round"] > fast["best_round"]

    def test_given_log_loss_boosted_trees_classify_held_out_moons_well(self):
        X, y, X_val, y_val = noisy_moons()
        model = GradientBoosting(n_rounds=60, learning_rate=0.5, max_depth=2, loss="log").fit(X, y)
        assert np.mean(model.predict(X_val) == y_val) > 0.85


class TestWhenTreesWin:
    def test_given_mixed_tabular_features_a_tree_ensemble_beats_even_the_best_prepared_small_neural_network(self):
        scores = tabular_showdown()
        best_net = max(v for k, v in scores.items() if k.startswith("neural net"))
        assert min(scores["random forest"], scores["gradient boosting"]) > best_net

    def test_given_raw_unscaled_features_the_neural_network_does_worse_than_with_scaled_ones(self):
        scores = tabular_showdown()
        assert scores["neural net (raw inputs)"] < scores["neural net (scaled inputs)"]

    def test_given_log_income_and_one_hot_regions_the_neural_network_improves_on_scaling_alone(self):
        scores = tabular_showdown()
        assert scores["neural net (scaled + engineered)"] > scores["neural net (scaled inputs)"]

    def test_given_engineered_features_income_becomes_its_logarithm_and_each_region_its_own_column(self):
        # age, log income, late payments, six region flags, noise
        row = engineer_features(np.array([[30.0, 1000.0, 2.0, 4.0, 0.5]]))[0]
        assert row.tolist() == [30.0, pytest.approx(np.log(1000)), 2.0, 0, 0, 0, 0, 1, 0, 0.5]

    def test_given_a_house_bigger_than_any_seen_boosted_trees_predict_the_same_as_for_the_biggest_seen(self):
        # Every leaf holds an average of training prices, so predictions stay inside the range seen.
        model = GradientBoosting(n_rounds=20, learning_rate=0.5, max_depth=1).fit(SIZES, PRICES)
        assert model.predict(np.array([[40.0]]))[0] == model.predict(np.array([[4.0]]))[0]

    def test_given_the_tabular_data_a_feature_the_label_ignores_gets_the_least_permutation_importance(self):
        X, y, X_val, y_val, names = make_tabular()
        forest = RandomForest(n_trees=30).fit(X, y)
        drops = permutation_importance(forest, X_val, y_val)
        assert names[int(np.argmin(drops))] == "noise"

    def test_given_a_pure_noise_feature_impurity_importance_still_gives_it_credit(self):
        # Splits on noise still carve the training set, so impurity importance can't tell them from real ones.
        X, y, _, _, names = make_tabular()
        forest = RandomForest(n_trees=30).fit(X, y)
        assert forest.feature_importances[names.index("noise")] > 0.05
