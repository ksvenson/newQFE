
all: tests ads

.PHONY: clean tests ads

clean:
	$(MAKE) -C tests clean
	$(MAKE) -C ads_heatbath clean

examples:
	$(MAKE) -C tests

ads:
	$(MAKE) -C ads_heatbath
