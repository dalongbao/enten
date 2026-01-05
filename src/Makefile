EMCC = emcc
CFLAGS = -O2
SRC = main.c fish.c
OUT = build/index.html

all: $(OUT)

$(OUT): $(SRC) fish.h shell.html
	mkdir -p build
	$(EMCC) $(CFLAGS) -o $(OUT) $(SRC) -s USE_SDL=2 -s EXPORTED_FUNCTIONS='["_main","_set_action","_get_x","_get_y","_get_angle","_get_vx","_get_vy","_get_food_count"]' -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' --shell-file shell.html

clean:
	rm -rf build

.PHONY: all clean
