"""Interfaz de línea de comandos: ``uv run finnet <comando>``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from finnet.config import load_config

log = logging.getLogger("finnet")


def _cmd_sweep(args: argparse.Namespace) -> int:
    from finnet.experiments import run_sweep

    cfg = load_config(args.config)
    paths = run_sweep(
        cfg,
        out_dir=args.out,
        n_scenarios=args.scenarios,
        n_networks=args.networks,
        n_jobs=args.jobs,
        skip_nodes=args.skip_nodes,
        skip_scenarios=args.skip_scenarios,
    )
    for name, path in paths.items():
        log.info("%-10s -> %s", name, path)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from finnet.ml.train import train_all

    cfg = load_config(args.config)
    report = train_all(cfg, data_dir=args.data, models_dir=args.models)
    for name, m in report.items():
        log.info(
            "%s: %s",
            name,
            {k: round(v, 4) for k, v in m.items() if isinstance(v, float)},
        )
    return 0


def _cmd_world(args: argparse.Namespace) -> int:
    from finnet.world import generate_world

    world = generate_world(load_config(args.config), seed=args.seed)
    out = world.save(args.out)
    log.info(
        "Mundo (semilla %d): %d firmas, %d personas, %d relaciones -> %s",
        world.seed,
        int(world.is_firm.sum()),
        int(world.is_person.sum()),
        len(world.edges),
        out,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finnet",
        description="Riesgo sistémico en redes de pagos: experimentos y modelos.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Logging DEBUG")
    sub = parser.add_subparsers(dest="command", required=True)

    sw = sub.add_parser(
        "sweep", help="Barrido de escenarios -> tablas Parquet en data/"
    )
    sw.add_argument(
        "config", type=Path, help="YAML del barrido (p. ej. configs/sweep.yaml)"
    )
    sw.add_argument("--out", type=Path, default=None, help="Directorio de salida")
    sw.add_argument(
        "--scenarios", type=int, default=None, help="Sobrescribe scenarios.n"
    )
    sw.add_argument(
        "--networks", type=int, default=None, help="Sobrescribe pool.n_networks"
    )
    sw.add_argument(
        "--jobs", type=int, default=None, help="Procesos (por defecto: YAML)"
    )
    sw.add_argument(
        "--skip-nodes", action="store_true", help="No generar la tabla de nodos"
    )
    sw.add_argument(
        "--skip-scenarios",
        action="store_true",
        help="No generar la tabla de escenarios (p. ej. para regenerar solo nodos)",
    )
    sw.set_defaults(func=_cmd_sweep)

    tr = sub.add_parser("train", help="Entrena y versiona los modelos predictivos")
    tr.add_argument(
        "config", type=Path, help="YAML de entrenamiento (configs/train.yaml)"
    )
    tr.add_argument("--data", type=Path, default=Path("data"))
    tr.add_argument("--models", type=Path, default=Path("models"))
    tr.set_defaults(func=_cmd_train)

    wo = sub.add_parser(
        "world", help="Genera un mundo multicapa de clientes -> Parquet"
    )
    wo.add_argument("config", type=Path, help="YAML del mundo (configs/world.yaml)")
    wo.add_argument("--out", type=Path, default=Path("data/world/demo"))
    wo.add_argument("--seed", type=int, default=None, help="Sobrescribe la semilla")
    wo.set_defaults(func=_cmd_world)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("Interrumpido por el usuario")
        return 130


if __name__ == "__main__":
    sys.exit(main())
