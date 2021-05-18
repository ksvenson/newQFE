
all: src ads

.PHONY: clean ads src

clean:
	$(MAKE) -C src clean
	$(MAKE) -C ads_heatbath clean

src:
	$(MAKE) -C src

ads:
	$(MAKE) -C ads_heatbath
