#!/bin/bash
find $1 -name '*.pdb' | parallel --progress workflow/scripts/run_rinerator.sh {} $2
