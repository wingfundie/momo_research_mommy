import asyncio
from types import SimpleNamespace

from mom_break_bot import _dispatch_bare_command, _parse_bare_command


def test_parse_bare_command_with_arguments() -> None:
    assert _parse_bare_command("portfolio usd") == ("portfolio", ["usd"])
    assert _parse_bare_command("  MC sol legacy  ") == ("mc", ["sol", "legacy"])
    assert _parse_bare_command("performance 1y BTC") == (
        "performance",
        ["1y", "BTC"],
    )


def test_parse_bare_command_ignores_slash_commands_and_empty_text() -> None:
    assert _parse_bare_command("/portfolio usd") is None
    assert _parse_bare_command("   ") is None
    assert _parse_bare_command(None) is None


def test_dispatch_bare_command_routes_arguments_to_registered_handler() -> None:
    received = {}

    async def portfolio_handler(update, context) -> None:
        received["text"] = update.effective_message.text
        received["args"] = context.args

    update = SimpleNamespace(
        effective_message=SimpleNamespace(text="portfolio usd")
    )
    context = SimpleNamespace(
        application=SimpleNamespace(
            bot_data={"bare_command_handlers": {"portfolio": portfolio_handler}}
        ),
        args=None,
    )

    asyncio.run(_dispatch_bare_command(update, context))

    assert received == {"text": "portfolio usd", "args": ["usd"]}
