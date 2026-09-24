from elara.conversation import ConversationStore


def test_messages_state_and_clear(db):
    store = ConversationStore(db)
    cid = store.create()
    for i in range(5):
        store.add_message(cid, "user" if i % 2 == 0 else "assistant", f"m{i}", language="az")
    recent = store.recent_messages(cid, limit=3)
    assert [m.content for m in recent] == ["m2", "m3", "m4"]
    store.set_state(cid, {"references": [{"title": "x"}]})
    assert store.get_state(cid)["references"][0]["title"] == "x"
    store.clear(cid)
    assert store.recent_messages(cid) == [] and store.get_state(cid) == {}


def test_ensure_creates_unknown(db):
    store = ConversationStore(db)
    cid = store.ensure("does-not-exist")
    assert cid != "does-not-exist" and store.exists(cid)
    assert store.ensure(cid) == cid
