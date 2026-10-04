import argparse
from .server import create_app


def main():
    parser = argparse.ArgumentParser(description="Run the local world-model playground")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    create_app().run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
