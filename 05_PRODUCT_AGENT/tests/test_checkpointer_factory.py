from __future__ import annotations

import pytest

from agent.checkpointing import build_checkpointer
from api.settings import Settings


class FakeContext:
    def __init__(self, saver: "FakeSaver") -> None:
        self.saver = saver
        self.entered = False
        self.exited = False

    def __enter__(self) -> "FakeSaver":
        self.entered = True
        return self.saver

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.exited = True


class FakeSaver:
    last_url = ""
    last_context: FakeContext | None = None

    def __init__(self) -> None:
        self.setup_calls = 0

    @classmethod
    def from_conn_string(cls, url: str) -> FakeContext:
        cls.last_url = url
        cls.last_context = FakeContext(cls())
        return cls.last_context

    def setup(self) -> None:
        self.setup_calls += 1


def test_checkpointer_backend_none_returns_no_checkpointer():
    settings = Settings(_env_file=None, checkpointer_backend="none")

    assert build_checkpointer(settings) is None


def test_postgres_checkpointer_uses_database_url_and_runs_setup():
    settings = Settings(
        _env_file=None,
        checkpointer_backend="postgres",
        database_url="postgresql://customer_service:secret@postgres/customer_service",
    )

    checkpointer = build_checkpointer(settings, postgres_saver_cls=FakeSaver)

    assert FakeSaver.last_url == settings.database_url
    assert checkpointer.setup_calls == 1
    assert FakeSaver.last_context is not None
    assert FakeSaver.last_context.entered is True


def test_redis_checkpointer_uses_redis_url_and_runs_setup():
    settings = Settings(
        _env_file=None,
        checkpointer_backend="redis",
        redis_url="redis://redis:6379/0",
    )

    checkpointer = build_checkpointer(settings, redis_saver_cls=FakeSaver)

    assert FakeSaver.last_url == settings.redis_url
    assert checkpointer.setup_calls == 1


def test_checkpointer_can_skip_setup_for_prepared_infra():
    settings = Settings(
        _env_file=None,
        checkpointer_backend="postgres",
        database_url="postgresql://customer_service:secret@postgres/customer_service",
        checkpointer_setup=False,
    )

    checkpointer = build_checkpointer(settings, postgres_saver_cls=FakeSaver)

    assert checkpointer.setup_calls == 0


def test_postgres_checkpointer_requires_database_url():
    settings = Settings(_env_file=None, checkpointer_backend="postgres", database_url="")

    with pytest.raises(ValueError, match="DATABASE_URL"):
        build_checkpointer(settings, postgres_saver_cls=FakeSaver)


def test_redis_checkpointer_requires_redis_url():
    settings = Settings(_env_file=None, checkpointer_backend="redis", redis_url="")

    with pytest.raises(ValueError, match="REDIS_URL"):
        build_checkpointer(settings, redis_saver_cls=FakeSaver)
