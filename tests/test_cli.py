from tram_genomics.cli import main


def test_cli_reports_version(capsys):
    try:
        main(["--version"])
    except SystemExit as error:
        assert error.code == 0
    assert "TRAM 1.0.0" in capsys.readouterr().out
