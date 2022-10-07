import os
import options
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser("train config")
    config = options.parse_common_args(parser).parse_args()
    print(config.seed)