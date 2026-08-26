from elara.memory.persistent import PersistentMemory
from elara.memory.profile import UserProfile


def test_persistent_memory_saves_and_loads_profile(tmp_path):
    path = tmp_path / "memory.json"
    memory = PersistentMemory(path)

    profile = UserProfile(
        name="Nurguldur",
        pending_name="Aylindir",
    )

    memory.save_profile(profile)

    loaded = memory.load_profile()

    assert loaded.name == "Nurguldur"
    assert loaded.pending_name == "Aylindir"


def test_persistent_memory_returns_empty_profile_when_file_missing(tmp_path):
    path = tmp_path / "missing.json"
    memory = PersistentMemory(path)

    profile = memory.load_profile()

    assert profile.name is None
    assert profile.pending_name is None


def test_persistent_memory_handles_invalid_json(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("not valid json", encoding="utf-8")

    memory = PersistentMemory(path)

    profile = memory.load_profile()

    assert profile.name is None
    assert profile.pending_name is None
