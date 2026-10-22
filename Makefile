# Autonomous Low-Gravity Rotary-Percussive Bite-Drilling System MBSE Pipeline
PYTHON := /Users/jay/anaconda3/envs/envs563/bin/python
QUARTO := QUARTO_PYTHON=$(PYTHON) /usr/local/bin/quarto

.PHONY: all dirs design solar physics telemetry cad renders drawings handout sync test docs serve clean

all: dirs design solar physics telemetry cad renders drawings handout sync docs test

dirs:
	mkdir -p exports/cad exports/data exports/renders exports/drawings exports/handout docs_src/assets/cad docs_src/assets/data \
		docs_src/assets/js docs_src/assets/renders docs_src/assets/drawings docs_src/assets/handout docs

design: dirs
	$(PYTHON) -m src.design

solar: design
	$(PYTHON) -m src.solar

physics: design
	$(PYTHON) -m src.terramechanics

telemetry: physics solar
	$(PYTHON) -m src.state_machine

cad: design
	$(PYTHON) -m src.cad_generator

# Portfolio stills (numba z-buffer over the GLBs), A3 drawings (build123d HLR) and the one-page PDF summary
renders: cad
	$(PYTHON) -m src.render

drawings: design
	$(PYTHON) -m src.drawings

handout: renders telemetry
	$(PYTHON) -m src.handout

sync:
	cp -f exports/cad/*.glb exports/cad/*.step exports/cad/*.stl docs_src/assets/cad/
	cp -f exports/data/*.json docs_src/assets/data/
	cp -f exports/data/*.csv docs_src/assets/data/
	cp -f exports/renders/*.png docs_src/assets/renders/
	cp -f exports/drawings/*.svg exports/drawings/*.pdf docs_src/assets/drawings/
	cp -f exports/handout/design_summary.pdf exports/handout/design_summary.png docs_src/assets/handout/

test:
	$(PYTHON) -m pytest tests/test_physics_and_cad.py -v

docs: sync
	$(QUARTO) render docs_src

# GLB models cannot load over file:// - preview the built site over HTTP
serve:
	$(PYTHON) -m http.server 8000 --directory docs

clean:
	rm -rf exports/cad/* exports/data/* exports/renders/* exports/drawings/* exports/handout/* docs_src/assets/cad/* \
		docs_src/assets/data/* docs_src/assets/renders/* docs_src/assets/drawings/* docs_src/assets/handout/* docs/*
