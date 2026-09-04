from tram_genomics.cli import build_parser, main


def test_cli_reports_version(capsys):
    try:
        main(["--version"])
    except SystemExit as error:
        assert error.code == 0
    assert "TRAM 1.1.0" in capsys.readouterr().out


def test_tune_accepts_hierarchical_selection_tolerances():
    arguments = build_parser().parse_args([
        "tune",
        "--matrix", "matrix.tsv",
        "--groups", "groups.tsv",
        "--group1", "Red",
        "--group2", "Fodder",
        "--output-prefix", "model",
        "--rounds", "20",
        "--roc-auc-tolerance", "0.002",
        "--log-loss-tolerance", "0.003",
    ])

    assert arguments.roc_auc_tolerance == 0.002
    assert arguments.log_loss_tolerance == 0.003
