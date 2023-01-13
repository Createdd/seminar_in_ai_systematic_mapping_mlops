#!/bin/sh

echo 'convert ipynb files'
jupytext --to jupyter_conversion//py:light **/*.ipynb