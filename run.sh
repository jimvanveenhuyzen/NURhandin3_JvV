#!/bin/bash

python3 NURhandin3.py > NURhandin3problem1a.txt
python3 NURhandin3.py > NURhandin3problem1b.txt

echo "Generating the pdf"

pdflatex handin3_JvV_s2272881.tex
pdflatex handin3_JvV_s2272881.tex


