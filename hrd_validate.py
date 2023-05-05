import copy
import random
import time
import queue
import sys
import os
from queue import PriorityQueue

# Validation script for CSC384 A1 : hrd.py
# Change input0.txt, output0.txt and solution0.txt to run
#    other test inputs.
# Note: This only tests the A* solution. There are too many
#       possibly DFS solutions to check for an exact match.

if __name__ == '__main__':
  # Invoke the shell command to test the HRD solver
  print("Input file: hrd_input0.txt, output file: hrd_output0.txt")
  os.system("python3 hrd.py hrd_input2.txt hrd_dfs.txt hrd_output0.txt")

  output_read = open("hrd_output0.txt", "r")
  solution_read = open("hrd_solution0.txt", "r")

  output_lines = output_read.readlines()
  solution_lines = solution_read.readlines()
  passed = True

  if output_lines[0].startswith("Cost of the solution"):
    for index in range(1, len(output_lines)):
      if output_lines[index] != solution_lines[index]:
        print(f"Line {index + 1}: "
                             f"Expected <{output_lines[index].strip()}> "
                             f"Encountered <{solution_lines[index].strip()}>\n")
        passed = False
        break
  else:
    print("First line mismatch. Expected: Cost of the solution, Encountered: " + output_lines[0])

  print("HRD output matches solution file.")
