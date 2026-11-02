.PHONY: test demo compile clean-demo

test:
	python -m unittest discover -s tests -v

compile:
	python -m compileall -q small_control_harness experiments tests

demo:
	bash scripts/run_demo.sh

clean-demo:
	rm -rf artifacts/demo
