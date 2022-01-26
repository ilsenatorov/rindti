#!/bin/bash
BN=$(basename ${1%.*})
rinerator $1 $2/$BN >> /dev/null 2>&1
