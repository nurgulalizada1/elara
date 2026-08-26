from elara.core.result import ElaraResult


def test_elara_result_success():
    result = ElaraResult(
        success=True,
        message="Test successful",
    )

    assert result.success is True
    assert result.message == "Test successful"
    assert result.data is None


def test_elara_result_with_data():
    result = ElaraResult(
        success=True,
        message="Data loaded",
        data={"name": "Nurgul"},
    )

    assert result.data["name"] == "Nurgul"
