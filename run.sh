#!/bin/bash

wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m11.txt
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m12.txt
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m13.txt
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m14.txt
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m15.txt

python3 NURhandin3.py > NURhandin3problem1a.txt
python3 NURhandin3.py > NURhandin3problem1b.txt

echo "Generating the pdf"

pdflatex handin3_JvV_s2272881.tex
pdflatex handin3_JvV_s2272881.tex


