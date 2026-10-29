"""Public interface; help and dry runs do not import GPU libraries."""
import argparse
import json
from .common import models


def main(argv=None):
    parser = argparse.ArgumentParser(prog='copyloop')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('presets', help='Print the six frozen main-experiment presets')
    download = commands.add_parser('download', help='Download a pinned model')
    download.add_argument('--model', choices=models(), required=True)
    download.add_argument('--output', required=True)
    download.add_argument('--dry-run', action='store_true')
    prepare = commands.add_parser('prepare', help='Download and prepare main benchmark prompts')
    prepare.add_argument('--model', choices=models(), required=True)
    prepare.add_argument('--model-path', required=True)
    prepare.add_argument('--dataset', required=True)
    prepare.add_argument('--assets', default='data/assets')
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--offline', action='store_true')
    prepare.add_argument('--limit', type=int, help='Smoke-test subset; never reported as full benchmark')
    generate = commands.add_parser('generate', help='Launch one server and run a prepared benchmark')
    generate.add_argument('--prepared', required=True)
    generate.add_argument('--method', choices=['ar', 'copyloop'], default='copyloop')
    generate.add_argument('--output', required=True)
    generate.add_argument('--port', type=int, default=30000)
    generate.add_argument('--dry-run', action='store_true')
    evaluate = commands.add_parser('evaluate', help='Score completed generations')
    evaluate.add_argument('--run', required=True)
    summarize = commands.add_parser('summarize', help='Compare paired aggregate decode throughput')
    summarize.add_argument('--runs', nargs='+', required=True)
    summarize.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'presets':
        print(json.dumps(models(), indent=2))
    elif args.command in ('download', 'prepare'):
        from . import data
        getattr(data, args.command)(args)
    elif args.command == 'generate':
        from .runner import generate
        generate(args)
    elif args.command == 'evaluate':
        from .evaluation.run import evaluate
        evaluate(args)
    else:
        from .summary import summarize
        summarize(args)


if __name__ == '__main__':
    main()
