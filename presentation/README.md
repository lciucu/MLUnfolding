## Summary

Change your `LaTex` compiler to `LuaLaTex`, then compile `Presentation.tex`, which in turn will get the slies from `Talk.tex`. 

<meta http-equiv='Content-Type' content='text/html; charset=utf-8' />
<!-- markdownlint-disable MD003 MD033 MD034 -->

# DESY Beamer Style

This directory holds an adoption of DESY CD for presentations build in LaTeX
using Beamer.

It is assumed, that documents are processed using `lualatex`, which is available in current editions of TeXLive. These days this is the default TeX installation on most Linux platforms, it is pre installed on all DESY Linux Workstations as well as in the DESY TeX setup for other platforms. In case you have another edition of TeX installed you can get [TeXLive](https://www.tug.org/texlive/)

Apple users should use [MacTeX](https://www.tug.org/mactex/), essentially the built of TeXLive for Mac.


## Usage

This folder is meant to be self contained, you can install it into your `texmf-local` if you want to and it is assumed that if you want to you know how to do it.

For general usage, just copy the contents of this folder to a new folder for your talk (one folder per talk) and just edit away there. This has the main advantage that the whole presentation will work from within this very folder without any external prerequisits and it builds also a self contained archive safe version of your talk.

### Contents

#### The talk

Basically, building a talk is editing two files:

- `metadata.tex`: place all metadata here like your name, title of talk, date, keywords, abstract and so on. **You have to edit this file**
- `Talk.tex`: this is the actual talk. Don't get confused by `Presentation.tex`. This is just an infrastructure file including the `Talk.tex` to build the actual presentation alongside a printable handout. **This file holds the real content of the talk.**

The basic workflow is to add your TeX code to `Talk.tex` and call `make` or `make all` from the shell to build your talk and/or handout. If you do not have `make` available on your platform you can compile `Presentation.tex` for the slides and `Handout.tex` for the printable handout with your usual tools. However, please note that this style requires the modern `lualatex`-complier for proper utf-8 and font support. (It will not work with `latex` or `pdflatex`; `xelatex` could work, but is not tested.) If you happen to include a bibliography, it is suggested to use the modern utf-8 aware replacements `biblatex` and `biber` instead of the ancient `bibtex`.

Note: by default LaTeX Beamer uses a 4:3 aspect ratio common for ages for computer displays and beamers. If you are unsure what you get, this is usually a safe fall back that will display everywhere without loss of content. Modern equipment these days, however, often uses other aspect ratios like 16:9 or 16:10. So, this setup uses 16:9 as default. In case you do not have a 16:9 beamer check `Presentation.tex` to set a different aspect ratio as parameter to the `documentclass` statement.

Note: it is advisable to use _vector_ formats for graphics (e.g. pdf) and use relative `width` and `height` statements referring to `\textheight` and `\textwidth` (e.g. `width=0.5\textwidth` instead of `width=2cm`).

Some works about metadata (data describing your talk): It is important to set those values in `metadata.tex` for then they will be embedded in the PDF created. Search engines will extract them and use them for indexing. Some literature management tools (e.g. JabRef) can even extract them from the PDF and build the core of a bibliographic reference out of them. Thus, it is time well spent to add suitable keywords and your abstract here even though they will not show in your presentation.

Some lesser known fields:

- [ORCiD](https://orcid.org): basically, as a scientist you should get an ORCiD and also pass it to the library. ORCiD is an author identification system that allows to uniquely identify publications from one given author even if she has a very common name. ORCiD is more and more adopted by scientific publishers as well as authentication services. If you do not yet have an ORCiD registered, DESY encourages you to do so and give your ORCiD to the library. If you have questions concerning ORCiD feel free to contact the library via l.publishing@desy.de.
- DOI: if you publish your talk via DESY pubdb you will also get a DOI assigned to it and thus make your work citable and freely available. Adding the DOI here will also embed it in your PDF. To know the DOI that will be assigned upon release just submit your talk to the pubdb by creating a temp entry that you fill up later on. This will give you a PUDBD-Number and the DOI is formed appending the pubdb number to 10.3204/ (10.3204/PUBDB-yyyy-nnnn). For questions about this procedure feel free to contact l.publishing@desy.de
- Licence: You should always strive to make your work OpenAccess and specify a suitable licence. While logos for all CC licences are included and you are of course free to choose (also a completely different licence) DESY Library encourages to use CCBY for _scientific work_ to maximize re-usability. If you have questions concerning licensing your works feel free to ask your library via l.publishing@desy.de.

#### Infrastructure
- `Makefile`: the easiest way to compile your talk and generate a handout suitable for printing (collated pages 2 on 1) is just to run `make` from within the folder. `make` will then take care of the build process. It will also copy the resulting PDF files to the parent folder and move files created during compilation to `./tmp/` as a clean up measure.
  -`make` understands some paremeters:
    - `make all`: produces the Handout and the Presentation
    - `make clean`: remove files created thus starting the next run from scratch
    - `make Handout`: produce only the handout
    - `make Presentation`: produce only the presentation
  - If you happen to work on a Windows or Mac system without a working `make` you can of course compile manually:
    - To compile the presentation only: `lualatex Presentation`
    - To compile the handout only: `lualatex Handout`
- Beamer style files are all named `beamer*.sty`. You will _not_ want to touch those as they implement DESY CD.
- `macros.tex` is a helper file holding a bunch of macros (`\newcommand`s). It is also used to load all additional packages required. You may want to check this file and add just to see what's there already. Some of the commands defined might come in handy or give an idea.
- `img`: holds all artwork used for the style, e.g. all logos and stuff.
  - Note that the default assumes you're giving an english presentation, thus the english version of the HGF logo is included in the style files. You may however just replace the destination of the `HGF_Logo.pdf` link to point to the German version.
  - There are two versions of the DESY logo available:
    - `DESY_Logo.pdf`: the usual square logo of DESY, 2018 edition
    - `DESY_Word.pdf`: the DESY word mark for use within a text line (Note the macro `\DESYWord{}` to place this mark)
  - The [ORCiD](https://orcid.org) Logo. 
  - Several icons for [CreativeCommons Licences](http://creativecommons.org).

# Disclaimer

Copyright (c) 2018 DESY Library, approved by DESY PR 05/2018

This template is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this template; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA.

<!-- vim: spell spelllang=en_gb bomb
-->
