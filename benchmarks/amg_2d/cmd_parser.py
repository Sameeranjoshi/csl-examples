# Copyright 2024 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is not a real test, but a module that gets imported in other tests.

"""command parser for bandwidthTest
   
   --logsdir   working directory
   --driver     path to CSL compiler
   --fabric-dims  fabric dimension of a WSE
   --cmaddr       IP address of a WSE
   --channels        number of I/O channels, 1 <= channels <= 16
   --width-west-buf  number of columns of the buffer in the west of the core rectangle
   --width-east-buf  number of columns of the buffer in the east of the core rectangle
   --compile-only    compile ELFs
   --run-only        run the test with precompiled binary
"""


import os
import argparse


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--elffolder", 
      required=False, 
      default="out", help="prefix of ELF files")
  parser.add_argument(
      "--logsdir",
      help="folder to contain the log files (default: logs)")
  parser.add_argument(
      "--compile-only",
      help="Compile only", action="store_true")
  parser.add_argument(
      "--cmaddr",
      help="CM address and port, i.e. <IP>:<port>")
  parser.add_argument(
      "--run-only",
      help="Run only", action="store_true")
  
  args = parser.parse_args()

  logs_dir = "logs"
  if args.logsdir:
    logs_dir = args.logsdir

  dir_exist = os.path.isdir(logs_dir)
  if dir_exist:
    print(f"{logs_dir} already exists")
  else:
    print(f"create {logs_dir} to store log files")
    os.mkdir(logs_dir)

  return args, logs_dir
