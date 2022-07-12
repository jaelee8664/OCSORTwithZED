import argparse
from utils.args import make_parser
args = make_parser().parse_args()
print(args)
print(args.trt)