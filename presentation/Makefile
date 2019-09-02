TEX = 'lualatex'
BIB = 'biber'

all: Presentation.pdf
#all: Presentation.pdf Handout.pdf ../Presentation.pdf ../Handout.pdf

clean:
	rm -f Presentation.pdf Presentation.out.ps
	rm -f Handout.pdf Handout.out.ps 
	rm -f tmp/*
	rm -f *~

Presentation: Presentation.pdf

Handout: Handout.pdf

../Presentation.pdf: Presentation.pdf
	cp Presentation.pdf ..

../Handout.pdf: Handout.pdf
	cp Handout.pdf ..

Presentation.pdf: Presentation.tex Talk.tex myFrames.tex macros.tex metadata.tex beamercolorthemeDESY.sty beamerfontthemeDESY.sty beamerinnerthemeDESY.sty beamerouterthemeDESY.sty beamerthemeDESY.sty
	$(TEX) Presentation
	mv Presentation.aux tmp/
	mv Presentation.log tmp/
	mv Presentation.nav tmp/
	mv Presentation.snm tmp/
	mv Presentation.toc tmp/
	# mv Presentation.*.vrb tmp/

Handout.pdf: Handout.tex Talk.tex macros.tex metadata.tex beamercolorthemeDESY.sty beamerfontthemeDESY.sty beamerinnerthemeDESY.sty beamerouterthemeDESY.sty beamerthemeDESY.sty
	$(TEX) Handout
	mv Handout.aux tmp/
	mv Handout.log tmp/
	mv Handout.nav tmp/
	mv Handout.snm tmp/
	mv Handout.toc tmp/
	# mv Handout.*.vrb tmp/
	cp Handout.pdf ..
