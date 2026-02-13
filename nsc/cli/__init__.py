"""NSC CLI Package - Modular command structure for v1.1"""

import click

from nsc.cli.execute import execute_command
from nsc.cli.validate import validate_command
from nsc.cli.decode import decode_command
from nsc.cli.gate import gate_command
from nsc.cli.replay import replay_command
from nsc.cli.module import module_command
from nsc.cli.registry import registry_command


@click.group()
def main():
    """NSC Tokenless v1.1 CLI - Deterministic, auditable execution engine."""
    pass


main.add_command(execute_command, "execute")
main.add_command(validate_command, "validate")
main.add_command(decode_command, "decode")
main.add_command(gate_command, "gate")
main.add_command(replay_command, "replay")
main.add_command(module_command, "module")
main.add_command(registry_command, "registry")

__all__ = [
    "main",
    "execute_command",
    "validate_command", 
    "decode_command",
    "gate_command",
    "replay_command",
    "module_command",
    "registry_command",
]
