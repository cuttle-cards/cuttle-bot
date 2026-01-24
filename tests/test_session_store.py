import asyncio

import pytest

from server.session_store import SessionStore


@pytest.mark.asyncio
async def test_create_get_delete_session() -> None:
    store = SessionStore()

    session = await store.create_session(use_ai=False)
    fetched = await store.get_session(session.id)

    assert fetched is not None
    assert fetched.id == session.id
    assert await store.session_count() == 1

    deleted = await store.delete_session(session.id)
    assert deleted is True
    assert await store.get_session(session.id) is None
    assert await store.session_count() == 0


@pytest.mark.asyncio
async def test_concurrent_session_creation() -> None:
    store = SessionStore()

    tasks = [store.create_session(use_ai=False) for _ in range(10)]
    sessions = await asyncio.gather(*tasks)

    ids = {session.id for session in sessions}
    assert len(ids) == 10
    assert await store.session_count() == 10
