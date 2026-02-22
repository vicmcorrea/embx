from __future__ import annotations

import typer


def register_all_commands(app: typer.Typer, config_app: typer.Typer) -> None:
    from embx.commands.batch import register_batch_command
    from embx.commands.compare import register_compare_command
    from embx.commands.connect import register_connect_command
    from embx.commands.config import register_config_commands
    from embx.commands.doctor import register_doctor_command
    from embx.commands.embed import register_embed_command
    from embx.commands.models import register_models_command
    from embx.commands.ping import register_ping_command
    from embx.commands.providers import register_providers_command
    from embx.commands.quickstart import register_quickstart_command

    register_providers_command(app)
    register_models_command(app)
    register_connect_command(app)
    register_doctor_command(app)
    register_ping_command(app)
    register_quickstart_command(app)
    register_embed_command(app)
    register_batch_command(app)
    register_compare_command(app)
    register_config_commands(config_app)
