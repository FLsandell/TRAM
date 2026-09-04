import json
from pathlib import Path

import numpy as np
import pandas as pd
import tram_genomics.modeling as modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tram_genomics.modeling import (
    _snp_importances,
    _split_preencoded_features,
    train_replicates,
)
from tram_genomics.pipeline import run_pipeline
from tram_genomics.sliding_window import _annotate_genes, _read_gff, _window_sums, sliding_window_analysis


def test_modeling_and_sliding_window_smoke(tiny_data, tmp_path):
    output = tmp_path / "output"
    summary = train_replicates(
        matrix=tiny_data["matrix"],
        groups=tiny_data["groups"],
        target="SP_CODE",
        group1="Red",
        group2="Fodder",
        output=output,
        model=tiny_data["model"],
        replicates=2,
        jobs=1,
    )
    assert summary.exists()

    sliding_output = output / "sliding"
    threshold = sliding_window_analysis(
        summary=summary,
        chromosome=tiny_data["chromosome"],
        gff=tiny_data["gff"],
        database=tiny_data["database"],
        function=tiny_data["function"],
        output=sliding_output,
        repeat_fraction=0.1,
        randomizations=5,
        workers=1,
        window_size=200,
        step_size=100,
        significance_quantile=0.9,
        seed=7,
    )
    assert threshold >= 0
    assert (sliding_output / "windows_plot.png").exists()
    assert (sliding_output / "translated_go_terms.txt").exists()


def test_gff_reader_keeps_attributes_with_extra_tabs(tmp_path):
    gff = tmp_path / "genes.gff3"
    gff.write_text(
        "##gff-version 3\n"
        "Chr01\tphytozomev11\tgene\t706\t6715\t.\t-\t.\t"
        "ID=Phvul.001G000400\t.v2.1;Name=Phvul.001G000400\n",
        encoding="utf-8",
    )

    table = _read_gff(gff)

    assert table.loc[0, "seqid"] == "Chr01"
    assert table.loc[0, "TYPE"] == "gene"
    assert table.loc[0, "START"] == 706
    assert "ID=Phvul.001G000400" in table.loc[0, "attributes"]


def test_window_sums_clamps_tiny_negative_float_residue():
    positions = np.array([580, 1465, 1814, 2106, 2123, 2278, 2580, 3065, 3117, 3883])
    weights = np.array([
        0.0009172977047909027,
        0.00003959287666420286,
        0.0005285892632600216,
        0.00045933588288540373,
        0.00006234957914987561,
        0.000641328169139375,
        0.0008526328384806567,
        0.0005929410181042841,
        0.0002600974477372232,
        0.0008398815210314088,
    ])
    starts = np.arange(1, 5000, 5)

    sums = _window_sums(positions, weights, starts, 100)

    assert (sums >= 0).all()


def test_gene_annotation_includes_strongest_overlapping_window():
    windows = pd.DataFrame({
        "CHR": ["CHR1", "CHR1", "CHR1"],
        "seqid": ["scaffold_1", "scaffold_1", "scaffold_1"],
        "POS": [1, 100, 700],
        "VarImp_sum": [1.0, 2.0, 0.1],
    })
    gff = pd.DataFrame({
        "seqid": ["scaffold_1"],
        "TYPE": ["gene"],
        "START": [50],
        "STOP": [180],
        "attributes": ["ID=gene1"],
    })
    functions = pd.DataFrame({
        "#query": ["gene1.t1"],
        "Description": ["example one"],
        "GOs": ["GO:0000001"],
        "KEGG_ko": ["ko:K00001"],
        "PFAMs": ["PF00001"],
    })

    genes = _annotate_genes(windows, 0.5, gff, functions, window_size=200)

    assert len(genes) == 1
    assert genes.loc[0, "window_VarImp_sum"] == 2.0
    assert genes.loc[0, "window_start"] == 100
    assert genes.loc[0, "window_seqid"] == "scaffold_1"


def test_model_summary_counts_only_nonzero_importance_runs(tiny_data, tmp_path):
    model = tmp_path / "small_model.json"
    model.write_text(json.dumps({
        "n_estimators": 1,
        "max_depth": 1,
        "max_features": 1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "bootstrap": True,
    }), encoding="utf-8")
    summary = train_replicates(
        matrix=tiny_data["matrix"],
        groups=tiny_data["groups"],
        target="SP_CODE",
        group1="Red",
        group2="Fodder",
        output=tmp_path / "output",
        model=model,
        replicates=3,
        jobs=1,
    )
    varimp = pd.read_csv(summary.parent / "RF_VarImp_Red-Fodder.csv", sep="\t")
    table = pd.read_csv(summary, sep="\t")

    assert (varimp["VarImp"] > 0).all()
    assert table["VarImp_count"].max() <= 3
    assert table["VarImp_count"].min() < 3


def test_replicates_fit_one_hot_encoder_only_once(tiny_data, tmp_path, monkeypatch):
    original_encoder = modeling.OneHotEncoder

    class CountingEncoder(original_encoder):
        fit_transform_calls = 0

        def fit_transform(self, *args, **kwargs):
            type(self).fit_transform_calls += 1
            return super().fit_transform(*args, **kwargs)

    monkeypatch.setattr(modeling, "OneHotEncoder", CountingEncoder)

    train_replicates(
        matrix=tiny_data["matrix"],
        groups=tiny_data["groups"],
        target="SP_CODE",
        group1="Red",
        group2="Fodder",
        output=tmp_path / "output",
        model=tiny_data["model"],
        replicates=3,
        jobs=1,
    )

    assert CountingEncoder.fit_transform_calls == 1


def test_preencoded_replicate_matches_previous_training_fitted_encoding():
    features = np.array([
        [0, 0, -1, -128],
        [0, 0, 0, -128],
        [0, 1, 0, 127],
        [0, 1, 1, 127],
        [0, 0, 1, -128],
        [0, 1, 0, 127],
        [0, 0, 1, -128],
        [0, 1, 0, 127],
        [0, 2, -1, 127],
        [0, 2, 1, -128],
        [0, 2, 0, 127],
        [0, 2, 1, -128],
    ], dtype=np.int8)
    train_indices = np.arange(8)
    test_indices = np.arange(8, 12)
    train_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    previous_encoder = OneHotEncoder(handle_unknown="ignore")
    previous_train = previous_encoder.fit_transform(features[train_indices])
    previous_test = previous_encoder.transform(features[test_indices])

    global_encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    globally_encoded = global_encoder.fit_transform(features)
    full_category_counts = np.asarray(globally_encoded.getnnz(axis=0)).ravel()
    new_train, new_test, active_categories = _split_preencoded_features(
        globally_encoded,
        full_category_counts,
        train_indices,
        test_indices,
    )

    np.testing.assert_array_equal(new_train.toarray(), previous_train.toarray())
    np.testing.assert_array_equal(new_test.toarray(), previous_test.toarray())

    global_category_snps = np.repeat(
        np.arange(features.shape[1]),
        [len(categories) for categories in global_encoder.categories_],
    )
    previous_category_snps = np.repeat(
        np.arange(features.shape[1]),
        [len(categories) for categories in previous_encoder.categories_],
    )
    np.testing.assert_array_equal(
        global_category_snps[active_categories],
        previous_category_snps,
    )

    parameters = {
        "n_estimators": 20,
        "max_depth": 4,
        "max_features": "sqrt",
        "random_state": 31,
        "n_jobs": 1,
    }
    previous_model = RandomForestClassifier(**parameters).fit(
        previous_train,
        train_labels,
    )
    new_model = RandomForestClassifier(**parameters).fit(new_train, train_labels)
    np.testing.assert_array_equal(
        new_model.predict(new_test),
        previous_model.predict(previous_test),
    )
    np.testing.assert_array_equal(
        new_model.feature_importances_,
        previous_model.feature_importances_,
    )

    previous_offsets = np.cumsum(
        [0, *[len(categories) for categories in previous_encoder.categories_]]
    )
    previous_snp_importances = np.add.reduceat(
        previous_model.feature_importances_,
        previous_offsets[:-1],
    )
    new_snp_importances = _snp_importances(
        new_model,
        global_category_snps[active_categories],
        features.shape[1],
    )
    np.testing.assert_allclose(new_snp_importances, previous_snp_importances)


def test_index_split_preserves_previous_misclassification_order():
    features = pd.DataFrame(
        {"snp": np.arange(12)},
        index=[f"sample-{index}" for index in range(12)],
    )
    labels = np.array([0] * 6 + [1] * 6)
    old_train, old_test, old_y_train, old_y_test = train_test_split(
        features,
        labels,
        stratify=labels,
        test_size=0.25,
        random_state=37,
    )
    train_indices, test_indices, new_y_train, new_y_test = train_test_split(
        np.arange(len(features)),
        labels,
        stratify=labels,
        test_size=0.25,
        random_state=37,
    )

    assert old_train.index.equals(features.iloc[train_indices].index)
    assert old_test.index.equals(features.iloc[test_indices].index)
    np.testing.assert_array_equal(new_y_train, old_y_train)
    np.testing.assert_array_equal(new_y_test, old_y_test)

    incorrect_mask = np.array([True, False, True])
    assert old_test.loc[incorrect_mask].index.equals(
        features.iloc[test_indices[incorrect_mask]].index
    )


def test_complete_pipeline_writes_metadata(tiny_data, tmp_path):
    output = tmp_path / "complete"
    result = run_pipeline(
        matrix=tiny_data["matrix"],
        groups=tiny_data["groups"],
        target="SP_CODE",
        group1="Red",
        group2="Fodder",
        chromosome=tiny_data["chromosome"],
        gff=tiny_data["gff"],
        database=tiny_data["database"],
        function=tiny_data["function"],
        repeat_fraction=0.1,
        output=output,
        rounds=1,
        cv_folds=3,
        replicates=1,
        test_size=0.25,
        seed=7,
        jobs=1,
        randomizations=3,
        workers=1,
        window_size=200,
        step_size=100,
        significance_quantile=0.9,
        roc_auc_tolerance=0.002,
        log_loss_tolerance=0.003,
    )
    assert all(Path(path).exists() for path in result.values())
    assert (output / "run_metadata.json").exists()
    tuning = json.loads(Path(result["tuning"]).read_text(encoding="utf-8"))
    assert tuning["selection_policy"]["primary"] == "highest mean CV ROC-AUC"
    assert tuning["selection_policy"]["roc_auc_tolerance"] == 0.002
    assert tuning["selection_policy"]["log_loss_tolerance"] == 0.003
    assert tuning["selected_trial"]["mean_cv_roc_auc"] >= 0
    assert tuning["selected_trial"]["oof_log_loss"] >= 0
