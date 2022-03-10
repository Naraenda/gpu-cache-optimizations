import glob

bib_files = glob.glob(r'..\Papers\*.bib')

with open(r'bibliography.bib', 'w') as bibliography:
    for bib_file in bib_files:
        with open(bib_file, 'r') as bib:
            bibliography.write(bib.read())
            bibliography.write('\n\n')