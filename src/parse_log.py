# -*- coding: utf-8 -*-
import sys

if __name__ == "__main__":

    valid_losses = []
    valid_errors = []

    log_file = sys.argv[1]
    with open(log_file, "r") as file_ob:
        for line in file_ob:
            if "valid loss" in line:
                line_arr = line.strip().split(" ")
                r = float(line_arr[-1])
                valid_losses.append(r)
            if "valid error" in line:
                line_arr = line.strip().split(" ")
                r = float(line_arr[-1])
                valid_errors.append(r)
        print valid_losses
        print valid_errors
