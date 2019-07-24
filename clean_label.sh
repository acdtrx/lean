#!/bin/bash

echo "Removing ./runs/${1}-*"
rm -rf ./runs/${1}-*
echo "Removing ./output/${1}.txt"
rm ./output/${1}.txt