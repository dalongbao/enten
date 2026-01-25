CC = emcc
CFLAGS = -O2 -Wall
LDFLAGS = -s WASM=1 -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] -s ALLOW_MEMORY_GROWTH=1
SRCS = main.c fish.c simulator.c
OUT = build

all: $(OUT)/index.js

$(OUT)/index.js: $(SRCS) fish.h simulator.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(SRCS) -o $(OUT)/index.js $(LDFLAGS)
	cp index.html $(OUT)/
	-cp media/*.png $(OUT)/ 2>/dev/null || true

clean:
	rm -rf $(OUT)

.PHONY: all clean
