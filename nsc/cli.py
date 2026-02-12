"""NSC Tokenless v1.1 Command Line Interface."""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NSC Tokenless v1.1 - Tokenless execution engine with coherence governance"
    )
    
    # Global options
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--version", action="version", version="NSC Tokenless v1.1.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Execute command
    execute_parser = subparsers.add_parser("execute", help="Execute an NSC module")
    execute_parser.add_argument("module", type=Path, help="Path to NSC module JSON")
    execute_parser.add_argument("--registry", type=Path, help="Path to operator registry")
    execute_parser.set_defaults(func=_cmd_execute)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate an NSC module")
    validate_parser.add_argument("module", type=Path, help="Path to NSC module JSON")
    validate_parser.set_defaults(func=_cmd_validate)
    
    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode GLLL signal")
    decode_parser.add_argument("signal", type=str, help="JSON array of signal values")
    decode_parser.add_argument("--n", type=int, default=16, help="GLLL n parameter (default: 16)")
    decode_parser.set_defaults(func=_cmd_decode)
    
    # Gate command
    gate_parser = subparsers.add_parser("gate", help="Evaluate gate policy")
    gate_parser.add_argument("residuals", type=str, help="JSON object with residuals")
    gate_parser.set_defaults(func=_cmd_gate)
    
    # Replay command
    replay_parser = subparsers.add_parser("replay", help="Verify receipt chain through replay")
    replay_parser.add_argument("--receipts", type=Path, required=True, help="Path to receipts JSON")
    replay_parser.add_argument("--module", type=Path, help="Path to original module for verification")
    replay_parser.set_defaults(func=_cmd_replay)
    
    # Module command with subcommands
    module_parser = subparsers.add_parser("module", help="Module management commands")
    module_subparsers = module_parser.add_subparsers(dest="subcommand", help="Module subcommands")
    
    module_info_parser = module_subparsers.add_parser("info", help="Show module information")
    module_info_parser.add_argument("module", type=Path, help="Path to NSC module JSON")
    module_info_parser.set_defaults(func=_cmd_module_info)
    
    module_digest_parser = module_subparsers.add_parser("digest", help="Compute module digest")
    module_digest_parser.add_argument("module", type=Path, help="Path to NSC module JSON")
    module_digest_parser.set_defaults(func=_cmd_module_digest)
    
    module_link_parser = module_subparsers.add_parser("link", help="Link modules together")
    module_link_parser.add_argument("module", type=Path, help="Path to primary module JSON")
    module_link_parser.add_argument("--other", type=Path, action="append", help="Additional modules to link")
    module_link_parser.set_defaults(func=_cmd_module_link)
    
    # Registry command with subcommands
    registry_parser = subparsers.add_parser("registry", help="Registry management commands")
    registry_subparsers = registry_parser.add_subparsers(dest="subcommand", help="Registry subcommands")
    
    registry_list_parser = registry_subparsers.add_parser("list", help="List operators in registry")
    registry_list_parser.add_argument("--registry", type=Path, required=True, help="Path to registry JSON")
    registry_list_parser.set_defaults(func=_cmd_registry_list)
    
    registry_info_parser = registry_subparsers.add_parser("info", help="Show registry information")
    registry_info_parser.add_argument("--registry", type=Path, required=True, help="Path to registry JSON")
    registry_info_parser.set_defaults(func=_cmd_registry_info)
    
    registry_validate_parser = registry_subparsers.add_parser("validate", help="Validate registry structure")
    registry_validate_parser.add_argument("--registry", type=Path, required=True, help="Path to registry JSON")
    registry_validate_parser.set_defaults(func=_cmd_registry_validate)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    # Execute command
    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_execute(args):
    from nsc.cli.execute import execute_command
    return execute_command(args)


def _cmd_validate(args):
    from nsc.cli.validate import validate_command
    return validate_command(args)


def _cmd_decode(args):
    from nsc.cli.decode import decode_command
    return decode_command(args)


def _cmd_gate(args):
    from nsc.cli.gate import gate_command
    return gate_command(args)


def _cmd_replay(args):
    from nsc.cli.replay import replay_command
    return replay_command(args)


def _cmd_module_info(args):
    from nsc.cli.module import module_command
    args.subcommand = "info"
    return module_command(args)


def _cmd_module_digest(args):
    from nsc.cli.module import module_command
    args.subcommand = "digest"
    return module_command(args)


def _cmd_module_link(args):
    from nsc.cli.module import module_command
    args.subcommand = "link"
    return module_command(args)


def _cmd_registry_list(args):
    from nsc.cli.registry import registry_command
    args.subcommand = "list"
    return registry_command(args)


def _cmd_registry_info(args):
    from nsc.cli.registry import registry_command
    args.subcommand = "info"
    return registry_command(args)


def _cmd_registry_validate(args):
    from nsc.cli.registry import registry_command
    args.subcommand = "validate"
    return registry_command(args)


if __name__ == "__main__":
    sys.exit(main())
